from fastapi import HTTPException
import asyncio
from PIL import Image
import io
import aiohttp
import pyautogui
import asyncio
import base64


async def take_delayed_screenshot(delay: int) -> Image.Image:
    """Takes a screenshot after specified delay"""
    if delay > 0:
        await asyncio.sleep(delay)
    
    # Take screenshot using pyautogui (non-blocking as it's quick)
    screenshot = pyautogui.screenshot()
    return screenshot

async def crop_image(image: Image.Image, percent_each_side:int = 5) -> bytes:
    """
    Crops image to middle so that some extra things can be reoved and makes is easy for the OCR.  Converts cropped to bytes
    """
    width, height = image.size
    num = percent_each_side / 100
    
    # Calculate crop dimensions and Crop 12% from each side
    left = int(width * num)
    top = int(height * max(0.125, num)) # Top crops till my bookmark bar
    right = int(width * (1-num))
    bottom = int(height * (1-num))
    
    cropped = image.crop((left, top, right, bottom))
    
    img_byte_arr = io.BytesIO()
    cropped.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()


async def send_to_ocr(image_bytes: bytes, ocr_url: str) -> dict:
    """
    Sends image to OCR service
    """
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(ocr_url, json={'image_bytes': base64_str}, timeout=20) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=f"OCR service returned status {response.status}")
                return await response.json()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in getting OCR: {str(e)}")