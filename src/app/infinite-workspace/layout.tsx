import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "HVAC.AI - Infinite Workspace",
  description: "AI-powered HVAC blueprint analysis with infinite workspace",
};

export default function InfiniteWorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
