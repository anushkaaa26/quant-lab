import "./globals.css";

export const metadata = {
  title: "QuantLab Research Studio",
  description: "Quantitative validation & RL on Indian Equities",
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
