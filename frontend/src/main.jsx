import React from 'react';
import { createRoot } from 'react-dom/client';
import { MotionConfig } from 'framer-motion';
import App from './App';
import './styles/fonts.css';
import './styles/tokens.css';
import './styles/base.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode><MotionConfig reducedMotion="user"><App /></MotionConfig></React.StrictMode>,
);
