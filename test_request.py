import asyncio
import aiohttp
import numpy as np

import io
from PIL import Image
import os

async def test():
    async with aiohttp.ClientSession() as session:
        addr = "http://0.0.0.0:5000/api/test"

        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        np_array = np.random.randn(1, 100)
        async with session.post(addr, data=np_array.tostring(), headers=headers) as resp:
            print(resp.status)
            bytes = await resp.content.read(20000)
            print(bytes)
#            image = Image.open(io.BytesIO(bytes))
#            image.save(os.path.join('img_response', "1.png"))
            f = open(os.path.join('img_response', "1.png"), 'wb')
            f.write(bytearray(bytes))
            f.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(test())
