/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  
  // Enable Server Actions for Clerk
  experimental: {
    serverActions: true,
    staticPageGenerationTimeout: 120,
  },

};

module.exports = nextConfig;