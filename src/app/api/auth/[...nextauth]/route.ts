import NextAuth from "next-auth";
import type { NextAuthOptions } from 'next-auth';
import { authOptions } from '@/lib/auth';

// Use the shared authOptions from `src/lib/auth`. In this project we
// intentionally keep auth minimal/stubbed (no interactive signin UI).
const handler = NextAuth(authOptions as NextAuthOptions);
export { handler as GET, handler as POST };
