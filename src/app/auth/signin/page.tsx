import React from 'react';

// Placeholder signin page to satisfy Next.js type generation.
// The interactive signin UI is intentionally removed in this branch.
export default function SignInPage() {
  return (
    <div style={{padding:32, textAlign:'center'}}>
      <h2>Sign in is disabled</h2>
      <p>This project does not expose the signin UI in this build.</p>
    </div>
  );
}

export const runtime = 'edge';
// This file is no longer needed as the login/sign-in functionality has been removed.
// You can safely delete this file.
