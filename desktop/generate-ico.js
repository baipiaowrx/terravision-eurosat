const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const pngPath = path.join(__dirname, 'assets', 'icon.png');
const icoPath = path.join(__dirname, 'assets', 'icon.ico');

// Generate 256x256 PNG for ICO
sharp(pngPath)
  .resize(256, 256)
  .png()
  .toBuffer()
  .then(png256 => {
    // ICO header (6 bytes)
    const header = Buffer.alloc(6);
    header.writeUInt16LE(0, 0); // Reserved
    header.writeUInt16LE(1, 2); // Type: 1 = ICO
    header.writeUInt16LE(1, 4); // Number of images
    
    // ICO directory entry (16 bytes)
    const entry = Buffer.alloc(16);
    entry.writeUInt8(0, 0);    // Width (0 means 256)
    entry.writeUInt8(0, 1);    // Height (0 means 256)
    entry.writeUInt8(0, 2);    // Color palette
    entry.writeUInt8(0, 3);    // Reserved
    entry.writeUInt16LE(1, 4); // Color planes
    entry.writeUInt16LE(32, 6); // Bits per pixel
    entry.writeUInt32LE(png256.length, 8); // Size of image data
    entry.writeUInt32LE(22, 12); // Offset of image data (6 + 16 = 22)
    
    // Combine all parts
    const ico = Buffer.concat([header, entry, png256]);
    
    fs.writeFileSync(icoPath, ico);
    console.log('ICO icon generated:', icoPath);
  })
  .catch(err => {
    console.error('Error:', err);
  });