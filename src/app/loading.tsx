export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-4 px-4">
        <div className="mx-auto w-14 h-14 bg-primary rounded-lg flex items-center justify-center animate-pulse">
          {/* simple visual placeholder for server fallback */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="h-8 w-8 text-primary-foreground"
          >
            <path d="M12 2a1 1 0 011 1v2a1 1 0 11-2 0V3a1 1 0 011-1zM12 19a1 1 0 011 1v2a1 1 0 11-2 0v-2a1 1 0 011-1zM4.22 4.22a1 1 0 011.42 0l1.42 1.42a1 1 0 11-1.42 1.42L4.22 5.64a1 1 0 010-1.42zM17.94 17.94a1 1 0 011.42 0l1.42 1.42a1 1 0 11-1.42 1.42l-1.42-1.42a1 1 0 010-1.42zM2 12a1 1 0 011-1h2a1 1 0 110 2H3a1 1 0 01-1-1zM19 12a1 1 0 011-1h2a1 1 0 110 2h-2a1 1 0 01-1-1zM4.22 19.78a1 1 0 000-1.42l1.42-1.42a1 1 0 011.42 1.42L5.64 19.78a1 1 0 00-1.42 0zM17.94 6.06a1 1 0 000-1.42l1.42-1.42a1 1 0 011.42 1.42L19.36 6.06a1 1 0 00-1.42 0zM12 6a6 6 0 100 12 6 6 0 000-12z" />
          </svg>
        </div>

        <div className="space-y-2">
          <div className="mx-auto h-6 w-6 animate-spin rounded-full border-4 border-t-transparent border-muted-foreground"></div>
          <p className="text-sm text-muted-foreground">Loading HVACAI...</p>
        </div>
      </div>
    </div>
  );
}
