import asyncio
import aiohttp
import numpy as np

import io
from PIL import Image
import os

async def test():
    async with aiohttp.ClientSession() as session:
        addr = "http://gan-api-hack.herokuapp.com/api/test"
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        np_array = np.random.randn(1, 100)
        async with session.post(addr, data=np_array.tostring(), headers=headers) as resp:
            print(resp.status)
            bytes = await resp.content.read(10000)
            print(bytes)
            image = Image.open(io.BytesIO(bytes))
            image.save(os.path.join('img_response', "1.png"))

loop = asyncio.get_event_loop()
loop.run_until_complete(test())
