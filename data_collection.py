import os
import asyncio
import aiohttp
from typing import Dict, List
import logging
import random
from dotenv import load_dotenv#
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                            logger.warning(f"HTTP {response.staus}: {await response.text()}")
                            raise aiohttp.ClientError(f"HTTP {response.staus}")
            except Exception as e:
                if attempt == max_retries - 1: # Final attempt
                    logger.error(f"Final attrempt failed: {e}")
                    raise
                else:
                    wait_time = (attempt + 1) * 2 + random.uniform(0,1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}")
                    await asyncio.sleep(wait_time)



async def main():
    api = AstroidAPI()

    request_params = {
        'start_date': '2020-01-01',
        'end_date': '2020-01-07',
        'api_key': os.getenv('API_KEY') if os.getenv('API_KEY') else 'DEMO_KEY'
    }
    data, url, status = await api.make_request(url=api.base_url, params=request_params)
    
    with open(f"data/raw/astroid_{request_params['start_date']}.json", 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    asyncio.run(main())