import logging
import datetime

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)

def main():
    logging.info("Starting the application")

if __name__ == "__main__":
    main()
