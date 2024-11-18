import os
import time

import cv2 as cv
from loguru import logger
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import pytesseract
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class ImageClassifier():

    def __init__(self):
        pass

    def do_processing(self,  filepath) -> None:
        logger.info(f'Processing ({filepath})...')
        filepath = self._do_file_conversion_if_needed(filepath)
        if not filepath:
            logger.info(f'Bailing out with a filepath ({filepath})')
            return
        logger.info('Reading...')
        image = cv.imread(filepath)
        # Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply GaussianBlur to remove noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding for better text segmentation
        processed_image = cv.adaptiveThreshold(
            blurred, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            11, 2
        )

        # Debug
        logger.info('Debugging...')
        debug_filepath = self._get_debug_filepath(filepath)
        cv.imwrite(debug_filepath, processed_image)
        logger.info('Running OCR...')
        text = pytesseract.image_to_string(Image.open(debug_filepath), lang='kor')
        logger.info('Done')
        logger.info(text)

    def _do_file_conversion_if_needed(self, filepath) -> str:
        extension = self._get_file_extension(filepath)
        if extension == '.heic':
            filepath = self._create_png_from_heic(filepath)
            return filepath
        return filepath

    def _create_png_from_heic(self, filepath) -> str:
        image_heic = Image.open(filepath)
        logger.info(image_heic)
        basename_first_part = os.path.splitext(os.path.basename(filepath))[0]
        new_filepath = f'./data/tmp/{basename_first_part}.png'
        logger.info(new_filepath)
        image_heic.save(new_filepath, format=format('png'))
        logger.info('Saved')
        return new_filepath

    def _get_file_extension(self, filepath: str) -> str:
        """
        Returns the file extension from the given file path.
        * @author chatgpt-4. 2024.

        :param file_path: The file path as a string.
        :return: File extension as a string (e.g., '.txt'), or an empty string if there's no extension.
        """
        return os.path.splitext(os.path.basename(filepath))[1]

    def _get_debug_filepath(self, filepath: str) -> str:
        basename_first_part = os.path.splitext(os.path.basename(filepath))[0]
        debug_filepath = f'./data/tmp/{basename_first_part}_debug.jpeg'
        return debug_filepath


class CustomFileSystemEventHandler(FileSystemEventHandler):

    def __init__(self, image_classifier: ImageClassifier):
        self.image_classifier = image_classifier  # As a reference pointer

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type == 'created' and not event.is_directory:
            self.image_classifier.do_processing(event.src_path)


def do_main_loop():
    image_classifier = ImageClassifier()

    # Start watching a directory.
    # Pass `image_classifier` as a shared object.
    event_handler = CustomFileSystemEventHandler(image_classifier)
    observer = Observer()
    observer.schedule(event_handler, './data', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


def main():
    do_main_loop()


if __name__ == '__main__':
    main()
