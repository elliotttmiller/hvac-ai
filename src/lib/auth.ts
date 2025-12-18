// Minimal auth shim for server session usage in some API routes.
// Historically this repo exported `authOptions` for next-auth. Some routes
// still import it even if full auth isn't wired in this environment. Provide
// a tiny stub so TS and runtime imports don't fail.

export const authOptions = {
	// provider-specific options would go here if enabled. Keep minimal.
	session: {
		strategy: 'jwt'
	}
};

// NOTE: If you use next-auth in production, replace this stub with the
// real configuration and remove the comment above.
