Primarily in `analyze_image_sequence.py`, this repository provides functions supporting a workflow for building language-study [Anki](https://apps.ankiweb.net/) digital flash cards:

1. Begins with an image sequence (provided by user; an example image sequence is provided in `test/sequence_sample_images.zip`), where each image follows a convention with filenames ending in `fname<FRAMENUM>.<FILETYPE>`
2. An OCR system (currently supported: Google Vision, EasyOCR) is invoked to extract text from each image in the sequence
3. A Japanese tokenizer is used to split the text into tokens (currently supported: JumanPP)
4. Online dictionaries/translators (currently supported: Jisho) are accessed to identify possible definitions for each token
5. The token/definition pairs are written to a text file easily parsed by Anki deck imports

It also provides optional functionality for building (very conservative) confidence intervals to estimate which of the available OCR systems has the
highest accuracy for the user's target image sequence.

The repository has a number of non-standard dependencies, notably:

1. [Fire](https://google.github.io/python-fire/guide/), for a convenient command-line interface, which allows functionality to be accessed like `python analyze_image_sequence.py FUNCTION_NAME --arg1=arg1val --arg2=arg2val`
2. [Python Google Vision API](https://codelabs.developers.google.com/codelabs/cloud-vision-api-python/#1), for use of Google's OCR neural nets (ala Google Lens). Because this is a commercial service, the user must also create a Google Cloud developer account, configure payment options, and obtain credentials, then modify `google_ocr.py` so that it knows where to find these. Concerning payment: as of May 11, 2024, the first 1000 uses per month of the Google Vision API are free, but pricing could of course change; anyone using the Google Vision API should consult [the Google Vision pricing page](https://cloud.google.com/vision/pricing) for current information
3. [EasyOCR](https://github.com/JaidedAI/EasyOCR), which provides a reasonably effective, free-and-open-source alternative to Google Vision. `analayze_image_sequence.py` can also use `EasyOCR` to filter images that likely contain no Japanese text, to reduce the number of calls to the Google Vision API
4. [JapaneseTokenizer](https://pypi.org/project/JapaneseTokenizer/), for convenient tokenizer interface, used in Step 3 (as well as JumanPP, for which the JapaneseTokenizer page provides installation instructions)
5. [cv2](https://pypi.org/project/opencv-python/), used in the optional CI estimation, to allow GUI user input on number of correct OCR extracts
6. [Noto Sans Japanese](https://fonts.google.com/noto/specimen/Noto+Sans+JP), the current font used in GUI display to the user when prompting for the number of correct OCR extracts, in building of confidence intervals. The relevant `.ttf` file is required in a folder within `fonts/`
