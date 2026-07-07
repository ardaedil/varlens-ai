import type { NextConfig } from "next"
import path from "node:path"

const nextConfig: NextConfig = {
  turbopack: {
    root: path.join(process.cwd(), "../.."),
  },
  transpilePackages: ["@varlens/contracts"],
}

export default nextConfig
