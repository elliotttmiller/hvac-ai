import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ClientBody from "@/components/layout/ClientBody";
import Script from "next/script";
import SessionProvider from "@/components/shared/SessionProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HVACAI - AI-Powered HVAC Management Platform",
  description:
    "Revolutionary HVAC project management through automated document processing, 2D-to-3D conversion, building codes compliance, and real-time project coordination.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <head>
        <Script
          crossOrigin="anonymous"
          src="//unpkg.com/same-runtime/dist/index.global.js"
        />
      </head>
      <body suppressHydrationWarning className="antialiased">
        <SessionProvider>
          <ClientBody>{children}</ClientBody>
        </SessionProvider>
      </body>
    </html>
  );
}

