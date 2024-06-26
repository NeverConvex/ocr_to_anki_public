# Non-standard modules
import numpy as np
import JapaneseTokenizer, fire, cv2, easyocr
from PIL import Image, ImageChops, ImageDraw, ImageFont

# TODO add fugashi support for tokenization, as well as CI estimation for tokenization correctness
"""
In [1]: import fugashi                                                                                                                                                                                                                                                                                                      In [2]: text = "麩菓子は、麩を主材料とした日本の菓子。"                                                                                                                                                                                                                                                                     In [3]: tagger = fugashi.Tagger()                                                                                                                                                                                                                                                                                           In [4]: words = [word.surface for word in tagger(text)]                                                                                                                                                                                                                                                                     In [5]: print(words)                                                                                                                                          ['麩', '菓子', 'は', '、', '麩', 'を', '主材', '料', 'と', 'し', 'た', '日本', 'の', '菓子', '。']
"""

# Standard modules
import time, os, pathlib, glob, string, json, path, warnings, random, shutil, decimal, importlib
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy

# Home-grown modules
import google_ocr
import chrome2anki_repo.util as c2a_util
all_ocrs = ["easyocr", "google.cloud.vision", "azure.ai.vision.imageanalysis"] # TODO add Tesseract, Texract support?
available_ocrs = [] # TODO use available_ocrs to check for invalid user args below and give helpful error messages
unavailable_ocrs = []
# General TODO: add Azure AI OCR support
# NOTE  AWS Textract reportedly works very well, but does not support Japanese as of 6/18/24:
#           https://docs.aws.amazon.com/textract/latest/dg/textract-best-practices.html#optimal-document
#       Tesseract does not seem to work well at all in practice without extensive image prep

# TODO deepDict/deepGet would be pretty natural developed into methods of a recursive defaultdict class (extending existing defaultdict?)
# TODO deepDict usage concept:
#   1. without tokenizing, take all 1-gram to 4-grams as candidate multi-keys
#   2. for any such k-gram that has a valid jisho entry, retain it as an actual multi-key
#   3. convert each character separately to a k, with "" used for <=3-grams, so a depth-5 deepDict is sufficient
#   4. for values, use structure like [(img_path # i, jisho entries # i), ...], which has en entry for each img with this k-gram in it
#   The motivation is that in searching OCR'd text to find an image pertinent to an entry manually entered into Jisho, we would normally
#   have to scan the list of images and use a list `in` for each image's OCR text. Repeating this for every search token would take
#   roughly #tokens * #imgs * #avg_ocr_text_len time
#   But with the deepDict, we can in constant time determine if the leading 4 characters of the search entry appear in any OCR text
#   (note: if we trusted tokenizers completely, we could use a deepDict with depth equal to the max len of a token to get a unique img path
#   in constant time)
#   This will likely do least well with e.g. verb conjugations
#   We may want to separately search on kana and kanji versions of each token, since we won't know which the dict is likely to have
def deepDict(cur_depth, default_val):
    """
    Builds arbitrary-depth defaultdicts.
    """
    if cur_depth > 1:
        return defaultdict(lambda: deepDict(cur_depth-1, default_val))
    return defaultdict(lambda: deepcopy(default_val))

def deepGet(inner_dict, keys):
    if len(keys)>1:
        return deepGet(inner_dict[keys[0]], keys[1:])
    return inner_dict[keys[0]]

def buildNGramDeepDict(context_img_json_paths, n=4, savepath=None, mode=None):
    """
    ngram_dict maps tokens, split along first 4 chars, to a list of (img_path, ocr_text) pairs, rpadded by blanks. e.g.,
    間違いcan be accessed like:
        ngram_dict["間"]["違"]["い"][""][img_path]
    and this returns the corresponding ocr_text in the image found at img_path.
    or {} iff no img_path was found to contain the partial string 間違い.
    """ 
    if savepath:
        if mode != "OVERWRITE":
            assert not os.path.isfile(savepath), f"{savepath} already exists, and mode is {mode}, not OVERWRITE."
    # TODO add a layer that tracks OCR system used as well? But we don't currently store this in the input JSONs; requires change elsewhere
    ngram_dict = deepDict(n, {})
    # context_img_jsons are assumed to map img_path -> ocr_text, as in the output from STEP 1 extractTextFromImgSequence below
    for jsp in context_img_json_paths:
         with open(jsp, 'r', encoding='utf8') as rf:
            ci_json = json.load(rf)
            for img_path, ocr_text in ci_json.items():
                for token_len in range(1, n+1):
                    for i in range(max(len(ocr_text) - token_len, 0) + 1):
                        igram = ocr_text[i:i+token_len]
                        if ' ' not in igram:
                            igram = igram.ljust(n) # Right-pad with ' ' to allow non-max-len tokens as uniform-depth dict keys
                            if '\n' not in igram and isCJK(igram[0]):
                                print(f"From {ocr_text}, start_pos {i} len {token_len} igram: {igram}")
                                deepGet(ngram_dict, igram[:-1])[igram[-1]][img_path] = ocr_text
    if savepath:
        with open(savepath, 'w', encoding='utf-8') as wf:
            json.dump(ngram_dict, wf, ensure_ascii=False)
    return ngram_dict

# TODO add checkAvailableTokenizers
def checkAvailableOCRs():
    for lib_name in all_ocrs:
        try:
            cur_lib = importlib.import_module(lib_name)
            available_ocrs.append(lib_name)
        except:
            unavailable_ocrs.append(lib_name)
    print(f"These OCRs appear to be available and properly configured (Python can import them): {available_ocrs}")
    print(f"Importing failed for these OCRs: {unavailable_ocrs}")

def resizeAllImages(input_folder, output_folder, downsize_factor=None, new_size=(None, None), img_type="jpeg"):
    """
    A helper fxn used outside the normal workflow. Storing full-resolution images in Anki media can be costly; this allows us to create mass-downscaled
    copies of images, without changing their names, so that we can copy the cheaper images over into Anki's media folder.
    """
    # NOTE: does not preserve image type, nor suffix, always forces output to jpg
    assert downsize_factor or new_size[0]
    input_img_paths = glob.glob(f"{input_folder}/*.{img_type}")
    print(f"resizeAllImages detected # {len(input_img_paths)} input images")
    for input_img_path in input_img_paths:
        img = Image.open(input_img_path)
        orig_w, orig_h = img.size
        if downsize_factor:
            new_size = (orig_w/downsize_factor, orig_h/downsize_factor)
        img.thumbnail(new_size, Image.Resampling.LANCZOS)
        img_basename, orig_ext = os.path.splitext(os.path.basename(input_img_path))
        img.save(f"{output_folder}/{img_basename}.jpg")

def copyOnlyFilesInAnkiImportable(path_to_importable, input_folder, output_folder):
    with open(path_to_importable, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    img_names = [l.strip().split('\t')[-1] for l in lines]
    for img_name in img_names:
        img_basename, orig_ext = os.path.splitext(img_name)
        shutil.copyfile(f"{input_folder}/{img_basename}.jpg", f"{output_folder}/{img_name}") # NOTE: .jpg assumes resized fxn was called

def easyOCRPicToText(img_path, verbose=True):
    reader = easyocr.Reader(['ja'])
    res = reader.readtext(img_path)
    ocr_text_extract = "" if len(res)==0 else res[0][1]
    if verbose:
        print(f"From {img_path}, EasyOCR extracted: {ocr_text_extract}")
    return ocr_text_extract

def ocr(img_path, ocr="google_ocr"):
    """
    Wrapper for generic OCR system access, to extract string text from img

    ocr options: google_ocr, easy_ocr
    """
    ocr_dict =  {
                    "google_ocr":google_ocr.pic_to_text,
                    "easy_ocr"  :easyOCRPicToText,
                }
    ocr_text_extract = ocr_dict[ocr](img_path)
    return ocr_text_extract

def vecDot(list1, list2):
    """
    numpy operations often run into overflow errors with large screenshots, so we implement this to work only with Python BigInts.
    """
    total = 0
    for val1, val2 in zip(list1, list2):
        total += val1 * val2
    return total

def imgDiff(im1, im2):
    if im1.size != im2.size:
        print(f"im1, im2 sizes non-identical: {im1.size}, {im2.size}")
        return 1.0
    diff_ub = int(np.product(im1.size)) * 3 * 255
    diff_vec    = ImageChops.difference(im1, im2).histogram() # NOTE this was np.array, but ran into overflow errors
    wt_vec      = list(range(256)) * 3
    normalized_diff = vecDot(diff_vec, wt_vec) / diff_ub
    assert 0. <= normalized_diff <= 1.0, f"Normalized diff outside [0, 1]: {normalized_diff}"
    return normalized_diff

def getFrameNum(impath, img_type):
    frame_num = int(impath[impath.rfind("frame")+len("frame"):impath.rfind(f".{img_type}")])
    return frame_num

def extractTokensFromHtmlLines(lines):
    google_jp_IME_space = "　"
    tokens = []
    for l in lines:
        if "jisho.org" in l and any(isCJK(c) for c in l):
            l = l.strip().replace(" - Jisho.org", '').replace("<DT>", '')
            l = l[l.find(">")+1:l.find("</A")]
            cur_tokens = l.split(google_jp_IME_space)
            print(f"In {l}, found cur_tokens: {cur_tokens}")
            tokens += cur_tokens
    print(f"Found {len(tokens)} tokens: {tokens}")
    return tokens

def isCJK(char):
    """
    Used to check for presence of at least one Japanese kanji/kana. See: https://stackoverflow.com/a/30070664/4286018
    This code is near-identical to the stack answer, with minor renaming for readability/avoiding reserved variable names

    See https://www.fileformat.info/info/unicode/block/index.htm It seems many Kanji are
    stored in unicode under a general "compatibility ideographs" (even if Kanji aren't exactly ideographs) label, with a few
    extra categories for hiragana, katana, and certain radicals (used e.g. as dictionary headers, presumably for kanji search).
    """
    # NOTE some argument in stack comments about whether this leaves a character off, maybe worth digging into TODO
    cjk_ranges = [
      {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},           # compatibility ideographs; https://en.wikipedia.org/wiki/CJK_Compatibility
      {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},           # compatibility ideographs; https://en.wikipedia.org/wiki/CJK_Compatibility_Forms
      {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},           # compatibility ideographs; https://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs
      {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},   # compatibility ideographs; https://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs_Supplement
      {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},           # Japanese Hiragana
      {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},           # Japanese Katakana
      {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},           # cjk radicals supplement
      # CJK Unified Ideographs; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
      {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},           
      # CJK Unified Ideographs Extension A; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A
      {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
      # CJK Unified Ideographs Extension B; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B
      {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
      # CJK Unified Ideographs Extension C; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_C
      {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
      # CJK Unified Ideographs Extension D; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_D
      {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
      # CJK Unified Ideographs Extension E; https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_E
      {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}
    ]
    return any([r["from"] <= ord(char) <= r["to"] for r in cjk_ranges])

def findSequenceUniques(target_folder="test/STEP0_sequence_sample_images_small", img_type="jpeg",
uniqueness_threshold=0.05, start_num=0, end_num=None, count_only=False):
    """
    Given a target folder ending in frame<num>, inspects all images with suffix img_type and removes those that appear to be near-unique,
    based on a normalized pixel-by-pixel abs diff compared to a threshold.
    """
    uniques_save_fname = "unique_saves/" + target_folder.replace("/", "_") + '.txt'
    img_paths = sorted(glob.glob(f"{target_folder}/*.{img_type}"), key=lambda impath: getFrameNum(impath, img_type))
    img_paths = img_paths if not end_num else img_paths[:end_num]
    img_paths = img_paths[start_num:]
    unique_img_paths = []
    total_num_imgs = len(img_paths)
    print(f"Total # images in {target_folder}: {total_num_imgs}")
    cur_img, cur_img_path, num_unique_imgs = None, None, 0

    if os.path.isfile(uniques_save_fname): # Check if we already analyzed this folder and saved the uniqueness results
        with open(uniques_save_fname, 'r') as rf:
            lines = rf.readlines()
            print(f"# lines in saved uniques file: {len(lines)}")
            if len(lines) == total_num_imgs:
                unique_img_paths = [l.strip() for l in lines if l.strip() != "NODIFF"]
                print(f"Total # images in {target_folder}: {total_num_imgs}")
                print(f"# unique imgs from file: {len(unique_img_paths)}")
                if not count_only:
                    return unique_img_paths
                return

    with open(uniques_save_fname, 'w') as wf:
        for i, img_path in enumerate(img_paths):
            print(f"Loading new image # {i + start_num} of {total_num_imgs + start_num} from: {img_path}")
            new_img = Image.open(img_path)
            if cur_img is None:
                unique_img_paths.append(img_path)
                wf.write(img_path + '\n')
                cur_img = new_img
                cur_img_path = img_path
                num_unique_imgs += 1
            else:
                img_diff_magnitude = imgDiff(cur_img, new_img)
                if img_diff_magnitude>uniqueness_threshold:
                    print(f"Detected image difference b/w {cur_img_path} and {img_path}: {img_diff_magnitude}")
                    cur_img = new_img
                    cur_img_path = img_path
                    num_unique_imgs += 1
                    unique_img_paths.append(img_path)
                    wf.write(img_path + '\n')
                else:
                    wf.write("NODIFF\n")
    print(f"Total # images in {target_folder}: {total_num_imgs}")
    print(f"# unique imgs: {num_unique_imgs} (this should match: {len(unique_img_paths)})")
    if not count_only:
        return unique_img_paths
    return

def removeImagesWithoutCJK(img_paths, filter_ocr):
    """
    Finds subset of img_paths for which filter_ocr OCR system extracts at least one CJK character
    """
    subset_img_paths = []
    for i, img_path in enumerate(img_paths):
        print(f"Requesting {filter_ocr} check for CJK text in img # {i} of {len(img_paths)}...")
        ocr_text_extract = ocr(img_path, ocr=filter_ocr)
        if any([isCJK(c) for c in ocr_text_extract]):
            subset_img_paths.append(img_path)
    return subset_img_paths

def extractTextFromImgSequence(target_folder="test/STEP0_sequence_sample_images_small", img_type="jpeg",
uniqueness_threshold=0.05, write_file="test/STEP1_sequence_sample_images_small_dump.txt", start_num=0, end_num=None, write_mode='a',
delay=1, ocr_system="google_ocr", filter_ocr=None):
    """
    STEP 1
    Examines unique (within threshold) images in target_folder, feeds them to a selected OCR system (currently only Google Vision supported; adding
    EasyOCR), extract texts from them, and stores text in write_file, as well as a mapping from img paths to text in <write_file w/o suffix>.json.
    """

    # TODO 1. add ability to use specified ocr to pre-filter images for the other, only retaining images that have detected CJK text
    # TODO this is especially useful if an open-source OCR (EasyOCR seems promising) is accurate enough to identify presence of text, but
    # TODO not accurate enough to correctly extract it, (but, alternatives like Tesseract don't even seem to detect text reliably)
    # TODO because these are images we don't have to feed to the Google Vision API, which costs both $$ and time.
    # TODO 2. Also allow detection of Latin alphabet, but our primary interest is in CJK, and specifically Japanese, characters
    img_paths = findSequenceUniques(  target_folder=target_folder, img_type=img_type, uniqueness_threshold=uniqueness_threshold)
                                        #start_num=start_num, end_num=end_num) # NOTE we think of #'ing as after uniqueness filter
    print(f"OCR sequence extract fxn found {len(img_paths)} files in target location {target_folder}")
    if filter_ocr:
        init_count = len(img_paths)
        img_paths = removeImagesWithoutCJK(img_paths, filter_ocr)
        print(f"After OCR filtering, of initial {init_count}, {len(img_paths)} remain in input image sequence")
    img_paths = img_paths if not end_num else img_paths[:end_num]
    img_paths = img_paths[start_num:]
    num_imgs = len(img_paths)

    img2text = {} # A dict mapping each filename to text OCR'd from it. Saved for reuse later to grab relevant imgs in CI estimation or Anki cards
    # TODO do we need to keep the wf writefile if we're also saving to img2text? img2text has strictly more information, but have to update later fxns..
    wf_name = write_file
    with open(wf_name, write_mode) as wf:
        for i, img_path in enumerate(img_paths):
            img_text = google_ocr.pic_to_text(img_path)
            img2text[img_path] = img_text
            time.sleep(delay) # A bit unclear if Google OCR intentionally throttles too-fast request rates..
            print(f"For img_path # {i + start_num} of {num_imgs + start_num}, Google Vision OCR returned: {img_text}")
            wf.write(f"{img_text}\n")
    wf_path_name, wf_ext = os.path.splitext(wf_name)
    json_wf_name = wf_path_name + ".json"
    with open(json_wf_name, 'w', encoding='utf-8') as wf:
        json.dump(img2text, wf, ensure_ascii=False)
    
    print(f"Done writing extracted text to:\n\t{wf_name}\n\t{json_wf_name}")

def allRomanOrNumber(sentence):
    return np.all([c in string.printable for c in sentence])

def tokenizeJP(read_file="test/STEP1_sequence_dump.json", write_file="test/STEP2_tokenized_sequence_dump.json"):
    """
    STEP 2
    Tokenizes lines in sequence_dump. Only tokenizes a line if it is detected as containing at least one CJK character.
    """
    assert read_file != write_file
    jpp = JapaneseTokenizer.JumanppWrapper()
    tokens_dict = {} # img_path -> (sentence, tokens map)
    #with open(read_file, 'r') as rf:
    #    input_lines = rf.readlines()        
    with open(read_file, 'r', encoding='utf8') as rf:
        input_json = json.load(rf)

    for i, (img_path, input_sentence) in enumerate(input_json.items()):
        input_sentence = input_sentence.strip()
        if any([isCJK(c) for c in input_sentence]):
            print(f"Tokenizing input sentence # {i} of {len(input_json)}: {input_sentence}")
            tok = jpp.tokenize(input_sentence) 
            tokens_as_list = tok.convert_list_object() 
            tokens_dict[img_path] = (input_sentence, tokens_as_list)
    with open(write_file, 'w', encoding='utf8') as wf:
        json.dump(tokens_dict, wf, ensure_ascii=False) 
    print(f"Wrote {len(tokens_dict)} tokenized lines to: {write_file}")

def getTokenTranslations( read_file="test/STEP2_tokenized_sequence_dump.json", method="Jisho", num_token_translations=None, delay=1,
write_file="test/STEP3_tokens_translated.json", context_img_json_glob_exprs=None):
    """
    STEP 3
    Looks up tokens in Jisho.

    If read_file
        : ends in .json:    - input tokens are directly read from JSON (assumed to come from previous OCR pipeline)
        : ends in .html:    - input tokens are parsed from HTML (assumed to come from Chrome bookmarks export)
                            - context_img_json_glob_exprs is used to find JSONs to search for relevant context images 
    """
    # TODO add DeepL, Google Translate, ChatGPT, etc as translation options?
    assert read_file != write_file

    read_basename, read_ext = os.path.splitext(os.path.basename(read_file))

    img2token2sentence2defns = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    tokens_queried = set()

    if read_ext == ".json": # Assumes json format as generated in STEP2
        with open(read_file, 'r', encoding='utf8') as rf:
            tokenized_json = json.load(rf)
        total_num_tokens = sum([len(sentence_tokens) for (input_sentence, sentence_tokens) in tokenized_json.values()])
        cur_token_index = 1
        for i, (img_path, (input_sentence, sentence_tokens)) in enumerate(tokenized_json.items()):
            for j, token in enumerate(sentence_tokens):
                if any([isCJK(c) for c in token]):
                    if token not in tokens_queried:
                        tokens_queried.add(token)
                        print(f"Looking for a definition for token # {cur_token_index} of {total_num_tokens}:  {token}")
                        jisho_guess = c2a_util.get_word_object(token)
                        if jisho_guess and 'data' in jisho_guess.keys() and jisho_guess['data']:
                            img2token2sentence2defns[img_path][token][input_sentence] = jisho_guess['data']
                        print(f"Jisho returned: {jisho_guess}")
                        time.sleep(delay) # Don't want to accidentally DDOS the Jisho folks
                cur_token_index += 1
                if num_token_translations and cur_token_index >= num_token_translations:
                    break
            if num_token_translations and cur_token_index >= num_token_translations:
                break
    elif read_ext == ".html": # Assumes bookmarks html format as exported from Chrome browser
        # Optionally, build deep default dict to search over for relevant context images
        token2img2ocr = None
        if context_img_json_glob_exprs:
            assert isinstance(context_img_json_glob_exprs, Iterable)
            context_img_json_paths = []
            for glob_expr in context_img_json_glob_exprs:
                context_img_json_paths += glob.glob(glob_expr)
            assert all([cijp.endswith(".json") for cijp in context_img_json_paths])
            print(f"Building {n}-gram DeepDict to find relevant images for context...")
            token2img2ocr = buildNGramDeepDict(context_img_json_paths, n=4, savepath=None, mode=None)

        # Search for token defns in jisho
        with open(read_file, 'r', encoding='utf8') as rf:
            lines = rf.readlines()
        tokens = extractTokensFromHtmlLines(lines)
        print(f"Found {len(tokens)} tokens in total")
        if num_token_translations:
            tokens = tokens[:num_token_translations]
        for cur_token_index, token in enumerate(tokens):
            if token not in tokens_queried:
                tokens_queried.add(token)
                print(f"Looking for a definition for token # {cur_token_index} of {len(tokens)}:  {token}")
                jisho_guess = c2a_util.get_word_object(token)
                if jisho_guess and 'data' in jisho_guess.keys() and jisho_guess['data']:
                    if token2img2ocr is None:
                        img2token2sentence2defns["None"][token]["None"] = jisho_guess['data']
                    else:
                        tk = token.ljust(4) # tk for 'token as key'
                        img_path_dict = token2img2ocr[tk[0]][tk[1]][tk[2]][tk[3]]
                        img_path, ocr_text = "None", "None"
                        for tmp_img_path, tmp_ocr_text in img_path_dict.items():
                            if token in tmp_ocr_text:
                                img_path, ocr_text = tmp_img_path, tmp_ocr_text
                                print(f"For {token}, found context img {img_path} with text: {ocr_text}")
                                break # Accept the first image we think contains this token
                                #(This generates a false positive if token is part of a larger word;
                                # but we can only fix that if we choose to trust a tokenizer, which would give false negatives. TODO implement 
                                # optional tokenizer support here, so we can easily compare the two approaches. Also consider a hybrid option:
                                # take first tokenizer match if any available, but if no tokenizer matches, take first string membership match.
                                # will yield exactly as many images as basic string membership approach, but should reduce false positives. May
                                # additionally add an option to limit this search to tokens containing kanji, or even to kanji-only tokens; 
                                # tokenization issues should be more limited especially with len>1 kanji-only tokens)
                        img2token2sentence2defns[img_path][token][ocr_text] = jisho_guess['data']
                print(f"Jisho returned: {jisho_guess}")
                time.sleep(delay) # Don't want to accidentally DDOS the Jisho folks
    else:
        raise ValueError(f"read-file extension: {read_ext} not recognized")

    with open(write_file, 'w', encoding='utf8') as of:
        json.dump(img2token2sentence2defns, of, ensure_ascii=False) 
    print(f"Wrote {len(img2token2sentence2defns)} tokens' translations to: {write_file}")

def generateAnkiImportableTxt(  read_file="test/STEP3_tokens_translated.json", write_file="test/STEP4_tokens_translated_anki_importable.txt",
filter_files=["auxiliary_inputs/wanikani_all_vocab.txt", "auxiliary_inputs/jlpt_N2_to_N5_notWK.txt"],
write_mode='a', respect_kana_only=True, include_context_img=True,
auxiliary_context_img_sources=[]):
    """
    STEP 4
    Assumes tab delimiters with fields in order:
        Expression Dfn Reading Grammar AddDefn Formality Tags [ImgName+suffix]
    filter_files are also assumed to be tab-delimiter; they will be checked for Expression, and Expression will only be added
    if it does not appear in any of the filter files.

    respect_kana_only: if jisho says the word is kana only, display reading as part of card expression
                (each card actually has several defns; we do this if the majority of them have 'kana only'; typically all-or-nothing)

    auxiliary_context_img_sources: list of strings passed as expressions to glob to determine a set of auxiliary JSONs. For tokens that do not have
            a context img_path, these JSONs will be searched for any sentence containing the token, and its img_path will be supplied
            as the context img_path here. Auxiliary JSONs are expected to be of the img2token2sentence2defns type produced in STEP3.
    """
    aux_src_img2token2sentence2defns = [] # TODO quadratic-time search for missing entries (slow). Reformat to char-by-char nested dict?
    if auxiliary_context_img_sources != []:
        aux_src_files = []
        for search_expr in auxiliary_context_img_sources:
            aux_src_files += glob.glob(search_expr)
        for aux_srcf in aux_src_files:
            with open(aux_srcf, 'r', encoding='utf8') as rf:
                ddict = json.load(rf)
                aux_src_img2token2sentence2defns.append(ddict)

    assert read_file != write_file
    # TODO consider option for adding the origin JP sentence itself (w/o image) as context to Anki card?
    filter_exprs = set()
    for fname in filter_files:
        with open(fname, 'r') as rf:
            lines = rf.readlines()
            for l in lines:
                if any([isCJK(c) for c in l]):
                    filter_exprs.add(l.strip())
    print(f"Received expression filter list with {len(filter_exprs)} entries")
    lines_written = 0
    written_exprs = set()
    with open(write_file, write_mode) as wf:
        with open(read_file, 'r', encoding='utf8') as rf:
            #img2token2sentence2defns[img_path][token][input_sentence] = jisho_guess['data']
            ddict = json.load(rf)
            for img_path in ddict.keys():
                for token in ddict[img_path].keys():
                    for sentence in ddict[img_path][token].keys(): # Should always just be one of these
                        jisho_dict = ddict[img_path][token][sentence][0] # First Jisho search result; currently ignore others
                        #print(jisho_dict['japanese'][0])
                        jp_keys = jisho_dict['japanese'][0].keys()
                        pos = [s['parts_of_speech'] for s in jisho_dict['senses']]
                        wiki_only = all(['Wikipedia definition' in p for p in pos])
                        #if 'word' in jp_keys and 'reading' in jp_keys: # TODO hm, are we excluding kana-only but actual words?
                        if 'reading' in jp_keys and not wiki_only:
                            reading = jisho_dict['japanese'][0]['reading']
                            kana_str = "Usually written using kana alone"
                            kana_only_bools = [kana_str in s['tags'] for s in jisho_dict['senses']]
                            #kana_only = round(sum(kana_only_bools)/len(kana_only_bools)) # Rounds 0.5 to 0; prefer to give tie to kana-only
                            # TODO remove wiki defn from consideration in performing kana-only calculation? Give preference to earlier defns?
                            kana_only = False
                            with decimal.localcontext() as ctx:
                                ctx.rounding = decimal.ROUND_HALF_UP # 0.5 rounds to 1
                                kana_only = int(decimal.Decimal(sum(kana_only_bools)/len(kana_only_bools)).to_integral_value())
                            # NOTE that this may not match the requested expression, since Jisho has interpreted (e.g., further tokenized) our request:
                            expr = base_expr = reading
                            if 'word' in jisho_dict['japanese'][0].keys():
                                expr = base_expr = jisho_dict['japanese'][0]['word']
                                if kana_only:
                                    expr = reading + ' (' + jisho_dict['japanese'][0]['word'] + ')'
                                    base_expr = reading
                            if expr not in written_exprs and base_expr not in filter_exprs:
                                written_exprs.add(expr)
                                eng_dfns = ['; '.join(s['english_definitions']) for s in jisho_dict['senses']]
                                eng_dfns = ' | '.join([f"{i}. {dfn_str}" for i, dfn_str in enumerate(eng_dfns)])
                                parts_of_speech = ['; '.join(s['parts_of_speech']) for s in jisho_dict['senses']]
                                parts_of_speech = ' | '.join([f"{i}. {pos_str}" for i, pos_str in enumerate(parts_of_speech)])
                                reading = jisho_dict['japanese'][0]['reading']
                                grammar = " "
                                adddefn = " "
                                formality = " "
                                tags = " "
                                write_vars = [expr, eng_dfns, parts_of_speech, reading, grammar, adddefn, formality, tags]
                                if include_context_img:
                                    written_img_path = img_path if img_path != "None" else ""
                                    write_vars.append(os.path.basename(written_img_path))
                                write_line = '\t'.join(write_vars) + '\n'
                                wf.write(write_line)
                                lines_written += 1
    print(f"Finished writing anki-importable version of token translations to: {write_file}")
    print(f"Total lines written: {lines_written}")

def estimateOCRCorrectness( ocr_system="google_ocr", ci_perc=0.95, sample_size=9, nrows=3, ncols=3, imgs_out_folder="test/imgs",
input_json_path=None, input_sequence_folder=None, img_type="jpg"):
    """
    OPTIONAL STEP [outside of normal workflow]

    Displays sample_size images drawn uniformly i.i.d. from input_sequence_folder or input_json_path, whichever is specified. Presents OCR-extracted
    text of these images to the user, prompting them to input indicate how many images had text correctly extracted. Modeling the percent of 
    correctly OCR'd images (from the set of those available locally) as the success probability of a binomial distribution, presents a
    ci_perc Clopper-Pearson confidence interval (exact, i.e., always achieves at least coverage probability, though can be quite conservative).
    Primary purpose is to support decision-making about whether alternative competing commercial (e.g., google_ocr, Azure AI OCR) systems
    are preferable to one another, or if an open-source alternative (e.g., easy_ocr, tesseract) may even be viable.

    Currently supported ocr_system options: None (but extractTextFromImgSequence can be used to generate a suitable JSON)
            (google_ocr, easy_ocr support to be added later, mimicking fxns above this one, primarily for small-scale testing)

    Note: input_json_path should be a json as output by STEP 1: extractTextFromImgSequence
    """
    assert 1<=sample_size
    assert nrows * ncols <= 9 # Need a more flexible user input system to support user grading of image grids with >=10 images
    assert 0 < ci_perc < 1.0
    # NOTE For bools (here, False usually comes from None or ''): (A xor B) iff (A != B)
    assert bool(input_json_path) != bool(input_sequence_folder), "Exactly one of input_json_path, input_sequence_folder is required, for sourcing samples"

    input_json = False
    if input_json_path:
        with open(input_json_path, 'r', encoding='utf8') as rf:
            input_json = json.load(rf)
    fnames_universe = list(input_json.keys()) if input_json_path else glob.glob(input_sequence_folder + f"*.{img_type}")
    all_sample_fnames = random.choices(fnames_universe, k=sample_size)

    samples_processed_ub = 0
    num_successes = 0
    while samples_processed_ub < sample_size:
        num_new_samples = min(sample_size - samples_processed_ub, nrows*ncols)
        fnames = all_sample_fnames[samples_processed_ub:samples_processed_ub + num_new_samples]
        print(f"Num new samples this iteration: {num_new_samples}")

        imgs = []
        for fname in fnames:
            print(f"Accessing {fname}...")
            img = Image.open(fname)
            
            # Resize image to be small enough that we can reasonably display it in a grid
            orig_w, orig_h = 3840, 2160 # Assumed monitor size TODO grab automatically
            resize_factor = 8
            resize_w, resize_h = int(orig_w/4), int(orig_h/4) # TODO check if non-integer? Though maybe tiny change in aspect ratio is imperceptible
            img.thumbnail((resize_w, resize_h), Image.Resampling.LANCZOS)
            img.save(f"{imgs_out_folder}/test_PIL_resize.jpg")

            # Insert whitespace below image
            right, left, top, bot = 0, 0, 0, 100
            tarea_resize_w, tarea_resize_h = resize_w, resize_h + bot
            tarea_resize_im = Image.new(img.mode, (tarea_resize_w, tarea_resize_h), (0, 0, 0))
            tarea_resize_im.paste(img, box=(left, top) )
            tarea_resize_im.save(f"{imgs_out_folder}/test_PIL_resize_whitespace.jpg")

            # Insert text in whitespace
            draw = ImageDraw.Draw(tarea_resize_im)
            #font = ImageFont.truetype("fonts/Times New Roman/times new roman.ttf", 16)
            #font = ImageFont.truetype("fonts/JayCons/Jaycons.ttf", 32)
            #font = ImageFont.truetype("fonts/KanjiStrokeOrdersMedium/KanjiStrokeOrders_v4.003.ttf", 32)
            font = ImageFont.truetype("fonts/Noto_Sans_JP/NotoSansJP-VariableFont_wght.ttf", 32)
            ocr_extract = input_json[fname] if input_json else ocr(fname, ocr=ocr_system)
            # TODO this currently can bleed out of the available drawspace:
            ocr_extract = ocr_extract.replace("\n", "") # TODO to fix, iteratively insert breaks to check/bound draw.textbbox text size
            print(f"ocr_extract for {fname}: {ocr_extract}")
            draw.text((0, resize_h + 0 * int(bot/2)), ocr_extract, (255,255,0), font=font)
            tarea_resize_im.save(f"{imgs_out_folder}/test_PIL_resize_whitespace_text.jpg")

            # Add modified image to list
            imgs.append(tarea_resize_im)

        # Simple PIL grid
        def pil_img_grid(imgs, nrows, ncols):
            print(f"Image grid received {len(imgs)} images. Attempting to display in {nrows} X {ncols} grid...")
            assert all(im.size == imgs[0].size for im in imgs), f"Cannot place imgs of uneven size in grid: {[im.size for im in imgs]}"
            if len(imgs) > nrows*ncols:
                warnings.warn(f"WARNING: {len(imgs)} provided; only displaying first {nrows*ncols}")
            elif len(imgs) < nrows*ncols:
                print(f"Received {len(imgs)}. Padding with blank images to reach {nrows*ncols}...")
                diff = nrows*ncols - len(imgs)
                blank_im = Image.new(imgs[0].mode, imgs[0].size, (0, 0, 0))
                imgs = imgs + [blank_im for _ in range(diff)]
            base_w, base_h = imgs[0].size
            grid = Image.new('RGB', size=(ncols*base_w, nrows*base_h))
            grid_w, grid_h = grid.size
            for row_num in range(nrows):
                for col_num in range(ncols):
                    img = imgs[row_num * ncols + col_num] # Fills out grid row-wise: along row 1, then along row 2, ..., left-to-right within row
                    grid.paste(img, box= (col_num*base_w, row_num*base_h) )
                    print(f"Pasting image # {row_num * ncols + col_num} of {len(imgs)} w/ box at: {(col_num*base_w, row_num*base_h)}")
            return grid
        imgrid = pil_img_grid(imgs, nrows, ncols)
        imgrid.save(f"{imgs_out_folder}/test_PIL_image_grid.jpg")

        # Insert whitespace for directions below grid image
        right, left, top, bot = 0, 0, 0, 100
        resize_w, resize_h = imgrid.size[0], imgrid.size[1] + bot
        imgrid_full = Image.new(imgrid.mode, (resize_w, resize_h), (0, 0, 0))
        imgrid_full.paste(imgrid, box=(left, top) )
        imgrid_full.save(f"{imgs_out_folder}/test_PIL_image_grid_whitespace.jpg")

        # Insert directions in grid whitespace # TODO update img to img list, img will become imgs[0]
        draw = ImageDraw.Draw(imgrid_full)
        font = ImageFont.truetype("fonts/Times New Roman/times new roman.ttf", 48)
        msg = "How many images have correctly extracted text? (tap a key in [1-9]; ESC for 0)"
        _, _, textw, texth = draw.textbbox((0, 0), msg, font=font)
        draw.text(( ((imgrid_full.size[0]-textw)/2), imgrid.size[1] + 0 * int(bot)), msg, (255,0,0), font=font)
        imgrid_full_save_loc = f"{imgs_out_folder}/test_PIL_image_grid_full.jpg"
        imgrid_full.save(imgrid_full_save_loc)

        display_img = cv2.imread(imgrid_full_save_loc)
        user_input_received = False
        while not user_input_received:
            cv2.namedWindow('OCR Sample', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("OCR Sample", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('OCR Sample', display_img)
            k = cv2.waitKey(33)
            for key_val in range(1, 9+1):
                if k==49-1+key_val: # System/platform-dependent? For me, 1-9 keys map to 49,...,57
                    num_successes += key_val
                    print(f"User indicated {key_val} successes (cv2 keyboard value: {k})")
                    user_input_received = True
                    break
                elif k==27: # Escape (0 seems to be -1, but this is also the default value, so not detectable by cv2?)
                    print(f"User indicated 0 successes (cv2 keyboard value: {k})")
                    user_input_received = True
                    break

        samples_processed_ub += nrows * ncols
        print(f"<={samples_processed_ub} of {sample_size} have been rated.")

    print(f"User completed rating sampled images: {num_successes} successes out of {sample_size} sampled images")

    #    Unpaywalled reference:    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    #           Source article:    https://academic.oup.com/biomet/article-abstract/26/4/404/291538?redirectedFrom=fulltext&login=false
    # NOTE Presumably due to how conservative Clopper-Pearson is, it requires large number of samples to get narrow CI. Examples:
    #           User completed rating sampled images: 42 successes out of 50 sampled images
    #               Point estimate of % correct OCRs: 0.84
    #               90.0th percentile confidence interval: (0.7297799224735874, 0.9178149382614393)
    #           User completed rating sampled images: 84 successes out of 100 sampled images
    #               Point estimate of % correct OCRs: 0.84
    #               90.0th percentile confidence interval: (0.7671841649934806, 0.8969888004843476)
    #           User completed rating sampled images: 161 successes out of 200 sampled images                                                         
    #               Point estimate of % correct OCRs: 0.805
    #               90.0th percentile confidence interval: (0.7531286899412225, 0.849953391358139)
    #           User completed rating sampled images: 315 successes out of 400 sampled images
    #               Point estimate of % correct OCRs: 0.7875
    #               90.0th percentile confidence interval: (0.7510665227225064, 0.8207277769606084)
    from scipy.stats import beta
    alpha = 1 - ci_perc # 1 - desired CI confidence
    p_lower, p_upper = beta.ppf([alpha/2, 1 - alpha/2], [num_successes, num_successes + 1], [sample_size - num_successes + 1, sample_size - num_successes])
    print(f"Point estimate of % correct OCRs: {num_successes/sample_size}")
    print(f"{100*ci_perc}th percentile confidence interval: ({p_lower}, {p_upper})")
    # TODO implement second, less conservative (but inexact) CI

def main():
    """
    Functions in this script are meant to be invoked from cmd-line via Fire. Simple example of processing pipeline, from OCR to Anki-importable txt:
    STEP 1:     python analyze_img_sequence.py extractTextFromImgSequence --filter_ocr="easy_ocr" --img_type="jpg"
    STEP 2:     python analyze_img_sequence.py tokenizeJP --read_file="test/STEP1_sequence_sample_images_small_dump.json" --write_file="test/STEP2_tokenized_sequence_sample_images_small_dump.json"
    STEP 3:     python analyze_img_sequence.py getTokenTranslations --read_file="test/STEP2_tokenized_sequence_sample_images_small_dump.json" --write_file="test/STEP3_tokens_translated_sample_images_small.json"
    STEP 4:     python analyze_img_sequence.py generateAnkiImportableTxt --read_file="test/STEP3_tokens_translated_sample_images_small.json" --write_file="test/STEP4_tokens_translated_anki_importable_sample_images_small.txt"

    NOTE: for images to work properly in Anki, the source images in the STEP 1 <target_folder> should be copied to %APPADATA%\Anki2\ for Windows, to
          ~/Library/Application/Support/Anki2/ for Mac, or to ~./local/share/Anki2/ for Linux. (These locations were taken from
          https://docs.ankiweb.net/files.html#file-locations as of May 2024, so if they do not seem to work, it is possible Anki updated its storage
          locations)
    """
    raise NotImplementedError(f"Do not run this script from main(). Call a fxn from cmd-line via Fire.")

if __name__ == "__main__":
    """
    Example run cmds:
    """
    checkAvailableOCRs()
    fire.Fire()
