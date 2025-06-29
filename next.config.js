/** @type {import('next').NextConfig} */
const nextConfig = {

  experimental: {
    serverActions: {
      allowedOrigins: ['localhost:3000', '127.0.0.1:3000','https://mira-aitutor.netlify.app', 'https://brave-rock-013225f00.2.azurestaticapps.net'],
    },
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://mira-backend-fcdndhgegjdhghf2.centralindia-01.azurewebsites.net/:path*',
      },
    ];
  },
};

module.exports = nextConfig;