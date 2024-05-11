This folder is used as a kind of cache to avoid unnecessarily repeating the uniques-filtering operation before OCR text extraction, which attempts
to limit OCR requests to only unique images. This operation takes a while, so we don't want to do it needlessly.
