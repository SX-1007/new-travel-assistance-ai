import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';

const authMocks = vi.hoisted(() => ({
  missingAuthConfig: [] as string[],
  listener: null as null | ((user: unknown) => void),
  watchSignedInUser: vi.fn(),
  currentIdToken: vi.fn(async () => 'id-token'),
  signInWithGoogle: vi.fn(async () => undefined),
  signInWithEmail: vi.fn(async () => undefined),
  createAccountWithEmail: vi.fn(async () => undefined),
  sendPasswordReset: vi.fn(async () => undefined),
  resendVerificationEmail: vi.fn(async () => undefined),
  reloadSignedInUser: vi.fn(async () => null as unknown),
  signOutUser: vi.fn(async () => undefined),
}));

const apiMocks = vi.hoisted(() => ({ setAuthTokenProvider: vi.fn() }));

vi.mock('../src/lib/auth', () => ({
  get missingAuthConfig() {
    return authMocks.missingAuthConfig;
  },
  watchSignedInUser: authMocks.watchSignedInUser,
  currentIdToken: authMocks.currentIdToken,
  signInWithGoogle: authMocks.signInWithGoogle,
  signInWithEmail: authMocks.signInWithEmail,
  createAccountWithEmail: authMocks.createAccountWithEmail,
  sendPasswordReset: authMocks.sendPasswordReset,
  resendVerificationEmail: authMocks.resendVerificationEmail,
  reloadSignedInUser: authMocks.reloadSignedInUser,
  signOutUser: authMocks.signOutUser,
  MIN_PASSWORD_LENGTH: 8,
}));

vi.mock('../src/lib/api', () => ({ setAuthTokenProvider: apiMocks.setAuthTokenProvider }));

import AuthGate, { useAuth } from '../src/app/components/AuthGate';

const ANA = { uid: 'uid-ana', displayName: 'Ana', email: 'ana@example.com', photoURL: null, needsEmailVerification: false };
const BEN = { uid: 'uid-ben', displayName: 'Ben', email: 'ben@example.com', photoURL: null, needsEmailVerification: false };
const CAI = { uid: 'uid-cai', displayName: 'cai@example.com', email: 'cai@example.com', photoURL: null, needsEmailVerification: true };

let mounts = 0;

function Probe() {
  const auth = useAuth();
  const [mountId] = React.useState(() => ++mounts);
  return (
    <div>
      <p>app for {auth?.user.email}</p>
      <p>mount {mountId}</p>
      <button type="button" onClick={auth?.signOut}>Sign out</button>
    </div>
  );
}


const emit = (user: unknown) => act(() => authMocks.listener?.(user));

const fillIn = (label: string, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe('sign-in gate', () => {
  beforeEach(() => {
    mounts = 0;
    authMocks.missingAuthConfig = [];
    authMocks.listener = null;
    Object.values(apiMocks).forEach((mock) => mock.mockReset());
    authMocks.watchSignedInUser.mockReset().mockImplementation((listener: (user: unknown) => void) => {
      authMocks.listener = listener;
      return () => { authMocks.listener = null; };
    });
    for (const mock of [
      authMocks.signInWithGoogle,
      authMocks.signInWithEmail,
      authMocks.createAccountWithEmail,
      authMocks.sendPasswordReset,
      authMocks.resendVerificationEmail,
      authMocks.signOutUser,
    ]) {
      mock.mockReset().mockResolvedValue(undefined);
    }
    authMocks.reloadSignedInUser.mockReset().mockResolvedValue(null);
  });

  it('waits for Firebase before showing anything', () => {
    render(<AuthGate><Probe /></AuthGate>);

    expect(screen.getByText(/Loading/)).toBeInTheDocument();
    expect(screen.queryByText(/app for/)).not.toBeInTheDocument();
  });

  it('shows only the sign-in screen when signed out, and signs in with Google', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);

    expect(screen.queryByText(/app for/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Continue with Google/ }));

    await waitFor(() => expect(authMocks.signInWithGoogle).toHaveBeenCalledOnce());
  });

  it('shows a sign-in failure the user can act on', async () => {
    authMocks.signInWithGoogle.mockRejectedValue(
      new Error('Your browser blocked the sign-in window. Allow pop-ups for this site and try again.'),
    );
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);

    fireEvent.click(screen.getByRole('button', { name: /Continue with Google/ }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Allow pop-ups');
  });

  it('renders the app for the signed-in user and hands the API its tokens', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(ANA);

    expect(screen.getByText('app for ana@example.com')).toBeInTheDocument();
    const provider = apiMocks.setAuthTokenProvider.mock.calls[0][0];
    await expect(provider.getToken(true)).resolves.toBe('id-token');
    expect(authMocks.currentIdToken).toHaveBeenCalledWith(true);

    provider.onRejected();
    expect(authMocks.signOutUser).toHaveBeenCalledOnce();
  });

  it('starts the app fresh when another account signs in', () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(ANA);
    expect(screen.getByText('mount 1')).toBeInTheDocument();

    emit(BEN);

    expect(screen.getByText('app for ben@example.com')).toBeInTheDocument();
    expect(screen.getByText('mount 2')).toBeInTheDocument();
  });

  it('signs out from inside the app', () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(ANA);

    fireEvent.click(screen.getByRole('button', { name: 'Sign out' }));

    expect(authMocks.signOutUser).toHaveBeenCalledOnce();
  });

  it('stops listening and forgets the token provider when unmounted', () => {
    const { unmount } = render(<AuthGate><Probe /></AuthGate>);

    unmount();

    expect(authMocks.listener).toBeNull();
    expect(apiMocks.setAuthTokenProvider).toHaveBeenLastCalledWith(null);
  });

  it('signs in with an email and password', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);

    fillIn('Email', 'ana@example.com');
    fillIn('Password', 'correct horse');
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    await waitFor(() => expect(authMocks.signInWithEmail).toHaveBeenCalledWith('ana@example.com', 'correct horse'));
  });

  it('shows a wrong password without saying whether the account exists', async () => {
    authMocks.signInWithEmail.mockRejectedValue(new Error('Incorrect email or password.'));
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);

    fillIn('Email', 'ana@example.com');
    fillIn('Password', 'wrong');
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect email or password.');
  });

  it('checks the new password before creating the account', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);
    fireEvent.click(screen.getByRole('button', { name: 'Create an account' }));
    fillIn('Email', 'cai@example.com');

    fillIn('Password', 'short');
    fillIn('Confirm password', 'short');
    fireEvent.click(screen.getByRole('button', { name: 'Create account' }));
    expect(screen.getByRole('alert')).toHaveTextContent('at least 8 characters');

    fillIn('Password', 'long enough 1');
    fillIn('Confirm password', 'long enough 2');
    fireEvent.click(screen.getByRole('button', { name: 'Create account' }));
    expect(screen.getByRole('alert')).toHaveTextContent('do not match');
    expect(authMocks.createAccountWithEmail).not.toHaveBeenCalled();

    fillIn('Confirm password', 'long enough 1');
    fireEvent.click(screen.getByRole('button', { name: 'Create account' }));
    await waitFor(() => expect(authMocks.createAccountWithEmail).toHaveBeenCalledWith('cai@example.com', 'long enough 1'));
  });

  it('sends a reset link with the same reply whether or not the account exists', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(null);
    fireEvent.click(screen.getByRole('button', { name: 'Forgot password?' }));

    expect(screen.queryByLabelText('Password')).not.toBeInTheDocument();
    fillIn('Email', 'ana@example.com');
    fireEvent.click(screen.getByRole('button', { name: 'Send reset link' }));

    expect(await screen.findByRole('status')).toHaveTextContent('If ana@example.com has an account');
    expect(authMocks.sendPasswordReset).toHaveBeenCalledWith('ana@example.com');
  });

  it('keeps an unverified email account out of the app', () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(CAI);

    expect(screen.getByRole('heading', { name: 'Verify your email' })).toBeInTheDocument();
    expect(screen.getByText('cai@example.com')).toBeInTheDocument();
    expect(screen.queryByText(/app for/)).not.toBeInTheDocument();
  });

  it('opens the app once the email is verified', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(CAI);

    authMocks.reloadSignedInUser.mockResolvedValueOnce(CAI);
    fireEvent.click(screen.getByRole('button', { name: "I've verified my email" }));
    expect(await screen.findByRole('alert')).toHaveTextContent("can't see the verification yet");

    authMocks.reloadSignedInUser.mockResolvedValueOnce({ ...CAI, needsEmailVerification: false });
    fireEvent.click(screen.getByRole('button', { name: "I've verified my email" }));
    expect(await screen.findByText('app for cai@example.com')).toBeInTheDocument();
  });

  it('checks again by itself when the user comes back from the email', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(CAI);

    authMocks.reloadSignedInUser.mockResolvedValueOnce({ ...CAI, needsEmailVerification: false });
    act(() => {
      window.dispatchEvent(new Event('focus'));
    });

    expect(await screen.findByText('app for cai@example.com')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('resends the verification link or lets the user switch account', async () => {
    render(<AuthGate><Probe /></AuthGate>);
    emit(CAI);

    fireEvent.click(screen.getByRole('button', { name: 'Send the link again' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Sent a new link to cai@example.com');

    fireEvent.click(screen.getByRole('button', { name: 'Use a different account' }));
    expect(authMocks.signOutUser).toHaveBeenCalledOnce();
  });

  it('explains missing configuration instead of failing silently', () => {
    authMocks.missingAuthConfig = ['VITE_FIREBASE_API_KEY', 'VITE_FIREBASE_APP_ID'];

    render(<AuthGate><Probe /></AuthGate>);

    expect(screen.getByRole('alert')).toHaveTextContent('Sign-in is not configured');
    expect(screen.getByText('VITE_FIREBASE_API_KEY')).toBeInTheDocument();
    expect(authMocks.watchSignedInUser).not.toHaveBeenCalled();
    expect(screen.queryByText(/app for/)).not.toBeInTheDocument();
  });
});

describe('sign-in error messages', () => {
  it('are written for the user, with a safe default', async () => {
    const actual = await vi.importActual<typeof import('../src/lib/auth')>('../src/lib/auth');

    expect(actual.signInErrorMessage('auth/popup-blocked')).toContain('Allow pop-ups');
    expect(actual.signInErrorMessage('auth/unauthorized-domain')).toContain('Authorized domains');
    expect(actual.signInErrorMessage('auth/invalid-credential')).toBe('Incorrect email or password.');
    expect(actual.signInErrorMessage('auth/email-already-in-use')).toContain('with Google');
    expect(actual.signInErrorMessage('auth/weak-password')).toContain('too weak');
    expect(actual.signInErrorMessage('auth/too-many-requests')).toContain('Too many attempts');
    expect(actual.signInErrorMessage('auth/something-new')).toBe('Sign-in failed. Please try again.');
  });
});
