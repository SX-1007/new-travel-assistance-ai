/**
 * auth.ts — sign-in through Firebase Authentication: Google, or an email and
 * password (which must be verified before the app opens).
 *
 * The only module that talks to Firebase. The API client asks it for an ID
 * token through `setAuthTokenProvider` (see api.ts), so the rest of the app
 * never imports Firebase.
 *
 * Config comes from Vite env vars (see .env.example). They are the public web
 * config of the Firebase project, not secrets:
 *   • VITE_FIREBASE_API_KEY
 *   • VITE_FIREBASE_AUTH_DOMAIN   — e.g. my-project.firebaseapp.com
 *   • VITE_FIREBASE_PROJECT_ID    — must match the backend's service account
 *   • VITE_FIREBASE_APP_ID
 */

import { initializeApp } from 'firebase/app';
import {
  EmailAuthProvider,
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  sendEmailVerification,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  type Auth,
  type User,
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY as string | undefined,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN as string | undefined,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID as string | undefined,
  appId: import.meta.env.VITE_FIREBASE_APP_ID as string | undefined,
};

/** Names of the sign-in settings that are missing; empty when configured. */
export const missingAuthConfig: string[] = Object.entries({
  VITE_FIREBASE_API_KEY: firebaseConfig.apiKey,
  VITE_FIREBASE_AUTH_DOMAIN: firebaseConfig.authDomain,
  VITE_FIREBASE_PROJECT_ID: firebaseConfig.projectId,
  VITE_FIREBASE_APP_ID: firebaseConfig.appId,
})
  .filter(([, value]) => !value?.trim())
  .map(([name]) => name);

let auth: Auth | null = null;

/** Created on first use, so importing this module needs no configuration. */
function firebaseAuth(): Auth {
  if (!auth) {
    auth = getAuth(initializeApp(firebaseConfig));
  }
  return auth;
}

export interface SignedInUser {
  uid: string;
  displayName: string;
  email: string;
  photoURL: string | null;
  /** An email-and-password account whose address isn't confirmed yet; the backend refuses it. */
  needsEmailVerification: boolean;
}

/** The shortest password the create-account form accepts (Firebase's own floor is 6). */
export const MIN_PASSWORD_LENGTH = 8;

const toSignedInUser = (user: User): SignedInUser => ({
  uid: user.uid,
  displayName: user.displayName ?? user.email ?? 'Signed in',
  email: user.email ?? '',
  photoURL: user.photoURL,
  needsEmailVerification:
    !user.emailVerified && user.providerData.some((info) => info.providerId === EmailAuthProvider.PROVIDER_ID),
});

/** Calls `listener` now and on every sign-in or sign-out; returns the unsubscribe. */
export function watchSignedInUser(listener: (user: SignedInUser | null) => void): () => void {
  return onAuthStateChanged(firebaseAuth(), (user) => listener(user ? toSignedInUser(user) : null));
}

/** The current user's ID token (refreshed by Firebase when near expiry), or null when signed out. */
export async function currentIdToken(forceRefresh: boolean): Promise<string | null> {
  const user = firebaseAuth().currentUser;
  return user ? user.getIdToken(forceRefresh) : null;
}

/** Resolves when signed in; resolves quietly when the user closes the popup. */
export async function signInWithGoogle(): Promise<void> {
  const provider = new GoogleAuthProvider();
  // Lets someone with several Google accounts pick one instead of reusing the last.
  provider.setCustomParameters({ prompt: 'select_account' });
  try {
    await signInWithPopup(firebaseAuth(), provider);
  } catch (error) {
    const code = errorCode(error);
    if (code === 'auth/popup-closed-by-user' || code === 'auth/cancelled-popup-request') return;
    throw new Error(signInErrorMessage(code));
  }
}

export async function signInWithEmail(email: string, password: string): Promise<void> {
  await withReadableErrors(() => signInWithEmailAndPassword(firebaseAuth(), email.trim(), password));
}

/**
 * Creates the account (which signs it in) and emails the verification link.
 * If that email fails, the account still exists; the verify screen can resend it.
 */
export async function createAccountWithEmail(email: string, password: string): Promise<void> {
  const credential = await withReadableErrors(() =>
    createUserWithEmailAndPassword(firebaseAuth(), email.trim(), password),
  );
  await sendEmailVerification(credential.user, backToApp()).catch(() => undefined);
}

export async function resendVerificationEmail(): Promise<void> {
  const user = firebaseAuth().currentUser;
  if (user) await withReadableErrors(() => sendEmailVerification(user, backToApp()));
}

/** Adds a "Continue" link on Firebase's "email verified" page back to this site (an authorised domain). */
const backToApp = () => ({ url: window.location.origin });

/**
 * Re-reads the account from Firebase, for after the user opens the
 * verification link. Once verified, the ID token is refreshed so the next
 * API call carries the verified claim.
 */
export async function reloadSignedInUser(): Promise<SignedInUser | null> {
  const user = firebaseAuth().currentUser;
  if (!user) return null;

  await withReadableErrors(() => user.reload());
  if (user.emailVerified) await user.getIdToken(true);
  return toSignedInUser(user);
}

/**
 * Sends a reset link. Resolves the same way whether or not the address has an
 * account, so the form can't be used to find out who has signed up.
 */
export async function sendPasswordReset(email: string): Promise<void> {
  try {
    await sendPasswordResetEmail(firebaseAuth(), email.trim());
  } catch (error) {
    const code = errorCode(error);
    if (code === 'auth/user-not-found') return;
    throw new Error(signInErrorMessage(code));
  }
}

export function signOutUser(): Promise<void> {
  return signOut(firebaseAuth());
}

function errorCode(error: unknown): string {
  const code = (error as { code?: unknown })?.code;
  return typeof code === 'string' ? code : '';
}

async function withReadableErrors<T>(action: () => Promise<T>): Promise<T> {
  try {
    return await action();
  } catch (error) {
    throw new Error(signInErrorMessage(errorCode(error)));
  }
}

/** A sentence the sign-in screen can show for a Firebase error code. */
export function signInErrorMessage(code: string): string {
  switch (code) {
    // Firebase reports a wrong password and an unknown email the same way
    // (email enumeration protection), so the message doesn't say which.
    case 'auth/invalid-credential':
    case 'auth/invalid-login-credentials':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return 'Incorrect email or password.';
    case 'auth/email-already-in-use':
      return 'An account already uses this email. Sign in instead (with Google, if that is how you joined), or reset your password.';
    case 'auth/invalid-email':
      return 'Enter a valid email address.';
    case 'auth/missing-password':
      return 'Enter your password.';
    case 'auth/weak-password':
    case 'auth/password-does-not-meet-requirements':
      return 'That password is too weak. Use a longer one with a mix of letters, numbers and symbols.';
    case 'auth/too-many-requests':
      return 'Too many attempts. Wait a few minutes and try again.';
    case 'auth/popup-blocked':
      return 'Your browser blocked the sign-in window. Allow pop-ups for this site and try again.';
    case 'auth/network-request-failed':
      return 'Could not reach Google. Check your connection and try again.';
    case 'auth/unauthorized-domain':
      return 'This address is not allowed to sign in. Add it under Authorized domains in the Firebase console.';
    case 'auth/operation-not-allowed':
      return 'This sign-in method is not enabled for this Firebase project.';
    case 'auth/user-disabled':
      return 'This account has been disabled.';
    default:
      return 'Sign-in failed. Please try again.';
  }
}
