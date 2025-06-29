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
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://mira-backend-fcdndhgegjdhghf2.centralindia-01.azurewebsites.net/:path*',
      },
    ];
  },
  
  // Environment variable validation
  env: {
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || '',
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;