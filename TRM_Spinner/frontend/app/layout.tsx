import type { Metadata } from "next";
import { Caveat, Patrick_Hand } from "next/font/google";
import "./globals.css";

const caveat = Caveat({
  subsets: ["latin"],
  variable: "--font-caveat",
  display: "swap",
});

const patrickHand = Patrick_Hand({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-patrick-hand",
  display: "swap",
});

export const metadata: Metadata = {
  title: "TRM Spinner",
  description: "Train tiny recursive models on your data",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${caveat.variable} ${patrickHand.variable}`}>
      <body className="min-h-screen bg-gator-50 antialiased">{children}</body>
    </html>
  );
}
