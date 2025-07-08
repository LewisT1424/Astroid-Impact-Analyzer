import os
import asyncio
import aiohttp
from typing import Dict, List
import logging
import random
from dotenv import load_dotenv#
import json
from utils import create_date_range

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#Date Format Exception - Expected format (yyyy-mm-dd) - The Feed date limit is only 7 Days","request":"http://api.nasa.gov/rest/v1/feed?start_date=2010-04-30&end_date=2010-01-22

class AstroidAPI:
    def __init__(self):
        self.base_url = 'https://api.nasa.gov/neo/rest/v1/feed'
    
    async def make_request(self, url: str, params: Dict, max_retries: int=3) -> List:
        # Make attempts
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=120, connect=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url=url, params=params) as response:
                        print('-'*75)
                        logger.info(f"Params: {params}")
                        logger.info(f"Response URL: {response.url}")
                        logger.info(f"Response status code: {response.status}")

                        if response.status == 200:
                            data = await response.json(content_type=None)
                            return [data, response.url, response.status]

                        else:
                            logger.warning(f"HTTP {response.status}: {await response.text()}")
                            raise aiohttp.ClientError(f"HTTP {response.status}")
            except Exception as e:
                if attempt == max_retries - 1: # Final attempt
                    logger.error(f"Final attempt failed: {e}")
                    raise
                else:
                    wait_time = (attempt + 1) * 2 + random.uniform(0,1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}")
                    await asyncio.sleep(wait_time)

    async def make_single_request(self, params, idx, total):
        try:
            data, url, status = await self.make_request(url=self.base_url, params=params)
            logger.info("Data successfully requested")
            path = f"data/raw/astroid_{params['start_date']}_{params['end_date']}.json"
            with open(path, 'w') as f:
                json.dump(data, f)
                logger.info(f"Data saved to {idx+1}/{total}: {path}")

        except Exception as e:
            logger.error(f"Error processing request: {idx + 1}: {e}")
            

    async def make_batch_request(self, start_date: str, end_date: str, batch_size: int=5):
        dates = create_date_range(start_date=start_date, end_date=end_date)
        
        for i in range(0, len(dates) - 1, batch_size):
            batch = dates[i:i+batch_size]
            tasks = []

            for idx, date in enumerate(batch[:-1]):
                params = {
                    'start_date': date,
                    'end_date': batch[idx+1],
                    'api_key': os.getenv('API_KEY') if os.getenv('API_KEY') else 'DEMO_KEY'
                }
                tasks.append(self.make_single_request(params=params, idx=idx+i, total=len(dates)))

            # Process this batch
            await asyncio.gather(*tasks, return_exceptions=True)
            # Add small delay between batches
            await asyncio.sleep(0.1)




async def main():
    api = AstroidAPI()
    start_date = '2010-01-01'
    end_date = '2011-01-01'

    await api.make_batch_request(start_date=start_date, end_date=end_date)
    logger.info(f"All data stored in data/raw")




if __name__ == '__main__':
    asyncio.run(main())