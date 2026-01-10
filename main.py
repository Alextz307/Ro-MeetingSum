import argparse
import logging

from src.scrapers.cdep_scraper import CDEPScraper
from src.processing.cleaner import DataProcessor
from src.evaluation.evaluator import ModelEvaluator
from src.config import START_ID, END_ID, RAW_FILE, CLEAN_FILE, TEST_SIZE_DEFAULT

# Import training modules if needed
# from src.training.train_matchsum import train as train_extractive
# from src.training.train_abstractive import train_abstractive


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ro-MeetingSum Pipeline CLI")

    parser.add_argument("--scrape", action="store_true", help="Run the scraper")
    parser.add_argument("--clean", action="store_true", help="Run the data cleaner")
    parser.add_argument("--train-ext", action="store_true", help="Train Extractive Model")
    parser.add_argument("--train-abs", action="store_true", help="Train Abstractive Model")
    parser.add_argument("--evaluate", action="store_true", help="Run Model Comparison/Evaluation")
    
    parser.add_argument("--start-id", type=int, default=START_ID, help="Start ID for scraper")
    parser.add_argument("--end-id", type=int, default=END_ID, help="End ID for scraper")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE_DEFAULT, help="Number of items to evaluate")

    args = parser.parse_args()

    if args.scrape:
        scraper = CDEPScraper(output_dir="data/processed")
        scraper.run_batch(start_id=args.start_id, end_id=args.end_id)

    if args.clean:
        proc = DataProcessor()
        proc.process_file(str(RAW_FILE), str(CLEAN_FILE))

    if args.train_ext:
        # train_extractive()
        print("Training Extractive Model... (Uncomment import in main.py to enable)")
        pass

    if args.train_abs:
        # train_abstractive()
        print("Training Abstractive Model... (Uncomment import in main.py to enable)")
        pass

    if args.evaluate:
        evaluator = ModelEvaluator(test_size=args.test_size)
        evaluator.run()

    if not any([args.scrape, args.clean, args.train_ext, args.train_abs, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
