import { createRoot } from 'react-dom/client';
import App from './app/App.tsx';
import AuthGate from './app/components/AuthGate.tsx';
import './styles/index.css';

createRoot(document.getElementById('root')!).render(
  <AuthGate>
    <App />
  </AuthGate>,
);
