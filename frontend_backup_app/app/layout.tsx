import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "SeismoGAN - Seismic Data Augmentation",
    description: "Advanced GAN-based denoising for MEMS seismic sensors",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.className} min-h-screen bg-background antialiased text-foreground`}>
                {children}
            </body>
        </html>
    );
}
