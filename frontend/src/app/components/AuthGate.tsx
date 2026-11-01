import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2, Mail, Plane } from 'lucide-react';

import { setAuthTokenProvider } from '../../lib/api';
import {
  MIN_PASSWORD_LENGTH,
  createAccountWithEmail,
  currentIdToken,
  missingAuthConfig,
  reloadSignedInUser,
  resendVerificationEmail,
  sendPasswordReset,
  signInWithEmail,
  signInWithGoogle,
  signOutUser,
  watchSignedInUser,
  type SignedInUser,
} from '../../lib/auth';

interface AuthContextValue {
  user: SignedInUser;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** The signed-in user and sign-out, or null outside the gate (as in component tests). */
export function useAuth(): AuthContextValue | null {
  return useContext(AuthContext);
}

type AuthState =
  | { status: 'loading' }
  | { status: 'signed-out' }
  | { status: 'signed-in'; user: SignedInUser };

const toState = (user: SignedInUser | null): AuthState =>
  user ? { status: 'signed-in', user } : { status: 'signed-out' };

/**
 * Shows the sign-in screen until someone signs in (Google, or an email whose
 * address is verified), then renders the app for that user. The app is keyed
 * by the user id, so switching accounts starts it fresh instead of showing
 * the previous user's state.
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: 'loading' });
  const configured = missingAuthConfig.length === 0;

  const signOut = useCallback(() => {
    void signOutUser();
  }, []);

  const applyUser = useCallback((user: SignedInUser | null) => setState(toState(user)), []);

  useEffect(() => {
    if (!configured) return undefined;

    setAuthTokenProvider({ getToken: currentIdToken, onRejected: signOut });
    const unsubscribe = watchSignedInUser(applyUser);
    return () => {
      unsubscribe();
      setAuthTokenProvider(null);
    };
  }, [configured, signOut, applyUser]);

  if (!configured) return <AuthNotConfigured missing={missingAuthConfig} />;

  if (state.status === 'loading') {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-50 text-slate-500">
        <Loader2 size={20} className="mr-2 animate-spin" /> Loading…
      </div>
    );
  }

  if (state.status === 'signed-out') return <SignInScreen />;

  if (state.user.needsEmailVerification) {
    return (
      <VerifyEmailScreen email={state.user.email} onReloaded={applyUser} onSignOut={signOut} />
    );
  }

  return (
    <AuthContext.Provider value={{ user: state.user, signOut }}>
      <React.Fragment key={state.user.uid}>{children}</React.Fragment>
    </AuthContext.Provider>
  );
}

/** Runs one sign-in action at a time and keeps its error or success message. */
function useAuthAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    setError(null);
    setNotice(null);
  };

  return { busy, error, notice, run, clear, setError, setNotice };
}

type EmailMode = 'sign-in' | 'sign-up' | 'reset';

const HEADINGS: Record<EmailMode, string> = {
  'sign-in': 'Sign in with email',
  'sign-up': 'Create an account',
  reset: 'Reset your password',
};

const SUBMIT_LABELS: Record<EmailMode, string> = {
  'sign-in': 'Sign in',
  'sign-up': 'Create account',
  reset: 'Send reset link',
};

const INPUT_CLASS =
  'w-full rounded-xl border border-slate-300 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100';

function SignInScreen() {
  const [mode, setMode] = useState<EmailMode>('sign-in');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const { busy, error, notice, run, clear, setError, setNotice } = useAuthAction();

  const switchMode = (next: EmailMode) => {
    setMode(next);
    setPassword('');
    setConfirm('');
    clear();
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();

    if (mode === 'sign-up') {
      if (password.length < MIN_PASSWORD_LENGTH) {
        setError(`Use at least ${MIN_PASSWORD_LENGTH} characters for your password.`);
        return;
      }
      if (password !== confirm) {
        setError('The two passwords do not match.');
        return;
      }
      void run(() => createAccountWithEmail(email, password));
    } else if (mode === 'reset') {
      void run(async () => {
        await sendPasswordReset(email);
        setNotice(`If ${email.trim()} has an account, a reset link is on its way. Check your inbox.`);
      });
    } else {
      void run(() => signInWithEmail(email, password));
    }
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-slate-50 p-4">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 rotate-3 items-center justify-center rounded-2xl bg-blue-100 text-blue-600">
            <Plane size={28} className="fill-current" />
          </div>
          <h1 className="text-2xl font-bold text-slate-800">Welcome to Wander AI</h1>
          <p className="mt-1 text-sm text-slate-500">Sign in to plan trips and keep them across devices.</p>
        </div>

        <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          {mode !== 'reset' && (
            <>
              <button
                type="button"
                onClick={() => void run(signInWithGoogle)}
                disabled={busy}
                className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <GoogleMark />
                Continue with Google
              </button>

              <div className="flex items-center gap-3 text-xs text-slate-400">
                <span className="h-px flex-1 bg-slate-200" />
                or use your email
                <span className="h-px flex-1 bg-slate-200" />
              </div>
            </>
          )}

          <form onSubmit={submit} className="space-y-3" noValidate>
            <h2 className="text-sm font-semibold text-slate-700">{HEADINGS[mode]}</h2>
            {mode === 'reset' && (
              <p className="text-sm text-slate-500">
                Enter your account's email and we'll send you a link to set a new password.
              </p>
            )}

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600">Email</span>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className={INPUT_CLASS}
              />
            </label>

            {mode !== 'reset' && (
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-600">Password</span>
                <input
                  type="password"
                  autoComplete={mode === 'sign-up' ? 'new-password' : 'current-password'}
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className={INPUT_CLASS}
                />
              </label>
            )}

            {mode === 'sign-up' && (
              <div>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-slate-600">Confirm password</span>
                  <input
                    type="password"
                    autoComplete="new-password"
                    required
                    aria-describedby="password-rule"
                    value={confirm}
                    onChange={(event) => setConfirm(event.target.value)}
                    className={INPUT_CLASS}
                  />
                </label>
                <p id="password-rule" className="mt-1 text-xs text-slate-400">
                  At least {MIN_PASSWORD_LENGTH} characters.
                </p>
              </div>
            )}

            <button
              type="submit"
              disabled={busy || !email.trim() || (mode !== 'reset' && !password)}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {busy && <Loader2 size={16} className="animate-spin" />}
              {SUBMIT_LABELS[mode]}
            </button>
          </form>

          {error && <Message kind="error">{error}</Message>}
          {notice && <Message kind="success">{notice}</Message>}

          <div className="flex flex-wrap justify-between gap-2 text-sm">
            {mode === 'sign-in' ? (
              <>
                <LinkButton onClick={() => switchMode('sign-up')}>Create an account</LinkButton>
                <LinkButton onClick={() => switchMode('reset')}>Forgot password?</LinkButton>
              </>
            ) : (
              <LinkButton onClick={() => switchMode('sign-in')}>
                {mode === 'sign-up' ? 'Already have an account? Sign in' : 'Back to sign in'}
              </LinkButton>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Shown to an email account until its address is verified; the backend refuses it before then. */
function VerifyEmailScreen({
  email,
  onReloaded,
  onSignOut,
}: {
  email: string;
  onReloaded: (user: SignedInUser | null) => void;
  onSignOut: () => void;
}) {
  const { busy, error, notice, run, setError, setNotice } = useAuthAction();

  // The link is usually opened in another tab, so check again quietly when
  // the user comes back to this one.
  useEffect(() => {
    const onFocus = () => {
      reloadSignedInUser()
        .then((user) => {
          if (!user?.needsEmailVerification) onReloaded(user);
        })
        .catch(() => undefined);
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [onReloaded]);

  const checkVerified = () =>
    run(async () => {
      const user = await reloadSignedInUser();
      if (user?.needsEmailVerification) {
        setError("We can't see the verification yet. Open the link in the email, then try again.");
        return;
      }
      onReloaded(user);
    });

  const resend = () =>
    run(async () => {
      await resendVerificationEmail();
      setNotice(`Sent a new link to ${email}.`);
    });

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-slate-50 p-4">
      <div className="w-full max-w-md space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-100 text-blue-600">
          <Mail size={24} />
        </div>
        <h1 className="text-xl font-bold text-slate-800">Verify your email</h1>
        <p className="text-sm text-slate-600">
          We sent a verification link to <strong>{email}</strong>. Open it, then come back here. If it hasn't
          arrived, check your spam folder or send it again.
        </p>

        <button
          type="button"
          onClick={() => void checkVerified()}
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy && <Loader2 size={16} className="animate-spin" />}
          I've verified my email
        </button>

        {error && <Message kind="error">{error}</Message>}
        {notice && <Message kind="success">{notice}</Message>}

        <div className="flex flex-wrap justify-between gap-2 text-sm">
          <LinkButton onClick={() => void resend()} disabled={busy}>
            Send the link again
          </LinkButton>
          <LinkButton onClick={onSignOut}>Use a different account</LinkButton>
        </div>
      </div>
    </div>
  );
}

function Message({ kind, children }: { kind: 'error' | 'success'; children: React.ReactNode }) {
  if (kind === 'error') {
    return (
      <p role="alert" className="flex items-start gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-700">
        <AlertTriangle size={16} className="mt-0.5 shrink-0" />
        {children}
      </p>
    );
  }

  return (
    <p role="status" className="flex items-start gap-2 rounded-lg bg-green-50 p-3 text-sm text-green-700">
      <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
      {children}
    </p>
  );
}

function LinkButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="font-medium text-blue-600 hover:underline disabled:cursor-not-allowed disabled:opacity-60"
    >
      {children}
    </button>
  );
}

function AuthNotConfigured({ missing }: { missing: string[] }) {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-slate-50 p-4">
      <div role="alert" className="w-full max-w-md rounded-2xl border border-amber-200 bg-white p-6 shadow-sm">
        <h1 className="flex items-center gap-2 text-lg font-bold text-slate-800">
          <AlertTriangle size={18} className="text-amber-500" /> Sign-in is not configured
        </h1>
        <p className="mt-2 text-sm text-slate-600">
          Add these settings from your Firebase project's web app to <code>frontend/.env.local</code>, then restart
          the dev server:
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-5 font-mono text-xs text-slate-700">
          {missing.map((name) => (
            <li key={name}>{name}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Google's "G" mark, as its sign-in branding guidelines ask for. */
function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z" />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-7.9l-6.5 5C9.5 39.6 16.2 44 24 44z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 38.2 44 33 44 24c0-1.3-.1-2.4-.4-3.5z" />
    </svg>
  );
}
