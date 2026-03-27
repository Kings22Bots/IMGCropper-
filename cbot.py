import cv2
import os
import numpy as np
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ==========================================
# 1. LOAD FULL-BODY AI MODEL (MobileNet SSD)
# ==========================================
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ==========================================
# ==========================================
# 2. DYNAMIC CROP LOGIC (Strict 9:16 Ratio)
# ==========================================
def extract_people_dynamically(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    cropped_files = []
    
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        
        if class_id == 15 and confidence > 0.3:
            # Get the exact bounding box of the person
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            box_w = endX - startX
            box_h = endY - startY
            center_x = startX + (box_w / 2)
            center_y = startY + (box_h / 2)
            
            # --- THE FIX: Force Strict 9:16 Ratio ---
            # Start by fitting the height with a tiny 6% padding (head to toe)
            target_h = box_h * 1.06
            target_w = target_h * (9 / 16)
            
            # If the person is wide, recalculate based on their width so arms aren't cut off
            if target_w < box_w * 1.1: 
                target_w = box_w * 1.1
                target_h = target_w * (16 / 9)
                
            # Calculate the final coordinates from the center point
            crop_startX = int(center_x - (target_w / 2))
            crop_endX = int(center_x + (target_w / 2))
            crop_startY = int(center_y - (target_h / 2))
            crop_endY = int(center_y + (target_h / 2))
            
            # Shift the box if it hits the edges of the photo to avoid black bars
            if crop_startX < 0:
                crop_endX += abs(crop_startX)
                crop_startX = 0
            if crop_endX > w:
                crop_startX -= (crop_endX - w)
                crop_endX = w
                
            if crop_startY < 0:
                crop_endY += abs(crop_startY)
                crop_startY = 0
            if crop_endY > h:
                crop_startY -= (crop_endY - h)
                crop_endY = h
                
            # Final safety clamp
            crop_startX = max(0, crop_startX)
            crop_startY = max(0, crop_startY)
            crop_endX = min(w, crop_endX)
            crop_endY = min(h, crop_endY)
            
            person_crop = image[crop_startY:crop_endY, crop_startX:crop_endX]
            
            output_filename = f"person_crop_{len(cropped_files)}.jpg"
            cv2.imwrite(output_filename, person_crop)
            cropped_files.append(output_filename)
            
    return cropped_files

# ==========================================
# 3. TELEGRAM HANDLERS
# ==========================================
async def handle_photo_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    await message.reply_text(" Image received! Scanning for people and framing...")
    
    if message.document:
        file_id = message.document.file_id
    else:
        file_id = message.photo[-1].file_id
        
    image_file = await context.bot.get_file(file_id)
    input_file = "temp_in.jpg"
    await image_file.download_to_drive(input_file)
    
    # Run our dynamic extraction
    extracted_people = extract_people_dynamically(input_file)
    
    if len(extracted_people) == 0:
        await message.reply_text("Could not clearly detect any people in this image to crop.")
    else:
        await message.reply_text(f" Found {len(extracted_people)} person(s)! Uploading spacious crops...")
        
        # Send all cropped people back as an album
        media_group = [InputMediaPhoto(open(file, 'rb')) for file in extracted_people]
        await message.reply_media_group(media=media_group)
        
        # Clean up the output files
        for file in extracted_people:
            if os.path.exists(file): os.remove(file)
            
    # Clean up input file
    if os.path.exists(input_file): os.remove(input_file)

# ==========================================
# 4. MAIN BOT SETUP
# ==========================================
if __name__ == '__main__':
    # ---> PASTE YOUR TELEGRAM TOKEN HERE <---
    TOKEN = os.environ.get("BOT_TOKEN")
    
    app = Application.builder().token(TOKEN).build()
    
    # Listen for compressed PHOTOS OR uncompressed image DOCUMENTS
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo_input))
    
    print("Bot is ready! Send an image. It will extract every person it finds with spacious padding.")
    app.run_polling()
