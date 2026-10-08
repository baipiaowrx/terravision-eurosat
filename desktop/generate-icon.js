const sharp = require('sharp');
const path = require('path');

const svgPath = path.join(__dirname, 'assets', 'icon.svg');
const pngPath = path.join(__dirname, 'assets', 'icon.png');

sharp(svgPath)
  .resize(256, 256)
  .png()
  .toFile(pngPath)
  .then(() => {
    console.log('PNG icon generated:', pngPath);
  })
  .catch(err => {
    console.error('Error:', err);
  });