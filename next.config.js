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
  
  // Exclude unnecessary files and folders from build
  outputFileTracingExcludes: {
    '*': [
      // Exclude API folder (FastAPI backend) - this is the main culprit
      'api/**/*',
      
      // Exclude test files
      '**/*.test.js',
      '**/*.test.ts',
      '**/*.test.jsx',
      '**/*.test.tsx',
      '**/*.spec.js',
      '**/*.spec.ts',
      '**/*.spec.jsx',
      '**/*.spec.tsx',
      
      // Exclude documentation and config files
      '**/*.md',
      '**/*.txt',
      '**/*.log',
      '**/*.yml',
      '**/*.yaml',
      '**/*.toml',
      '**/*.ini',
      
      // Exclude development tools
      '.eslintrc*',
      '.prettierrc*',
      'tsconfig.json',
      'postcss.config.js',
      'tailwind.config.ts',
      'next.config.js',
      
      // Exclude Docker and deployment files
      'Dockerfile*',
      'docker-compose*',
      '.dockerignore',
      '.github/**/*',
      
      // Exclude environment files
      '.env*',
      
      // Exclude IDE and editor files
      '.vscode/**/*',
      '.idea/**/*',
      '*.swp',
      '*.swo',
      '*~',
      
      // Exclude OS generated files
      '.DS_Store',
      'Thumbs.db',
      
      // Exclude uploads and temporary files
      'uploads/**/*',
      'vector_db/**/*',
      'temp/**/*',
      'tmp/**/*',
    ],
  },
};

module.exports = nextConfig;