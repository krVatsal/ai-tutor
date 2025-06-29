const fs = require('fs');
const path = require('path');

function getDirectorySize(dirPath) {
  let totalSize = 0;
  
  if (fs.existsSync(dirPath)) {
    const items = fs.readdirSync(dirPath);
    
    for (const item of items) {
      const itemPath = path.join(dirPath, item);
      const stats = fs.statSync(itemPath);
      
      if (stats.isDirectory()) {
        totalSize += getDirectorySize(itemPath);
      } else {
        totalSize += stats.size;
      }
    }
  }
  
  return totalSize;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

console.log('Build Size Analysis:');
console.log('===================');

const nextDir = path.join(__dirname, '.next');
if (fs.existsSync(nextDir)) {
  const totalSize = getDirectorySize(nextDir);
  console.log(`Total .next directory size: ${formatBytes(totalSize)}`);
  
  // Check specific directories
  const serverDir = path.join(nextDir, 'server');
  const staticDir = path.join(nextDir, 'static');
  
  if (fs.existsSync(serverDir)) {
    const serverSize = getDirectorySize(serverDir);
    console.log(`Server files size: ${formatBytes(serverSize)}`);
  }
  
  if (fs.existsSync(staticDir)) {
    const staticSize = getDirectorySize(staticDir);
    console.log(`Static files size: ${formatBytes(staticSize)}`);
  }
} else {
  console.log('No .next directory found. Run "npm run build" first.');
}

// Check if API folder exists and its size
const apiDir = path.join(__dirname, 'api');
if (fs.existsSync(apiDir)) {
  const apiSize = getDirectorySize(apiDir);
  console.log(`API folder size: ${formatBytes(apiSize)}`);
  console.log('⚠️  API folder should be excluded from frontend build!');
} else {
  console.log('API folder not found in current directory.');
} 