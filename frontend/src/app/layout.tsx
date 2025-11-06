import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DETECH - Decentralized AI Vigilance Swarm',
  description: 'Real-time environmental awareness with edge-swarm hybrid analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
