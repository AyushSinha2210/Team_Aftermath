import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Hero from './Hero';
import { shouldSkipIntro, useMotionPolicy } from '../hooks/useMotionPolicy';
import './laptop.css';

export default function LaptopReveal({ disabled = false }) {
  const reduced = useMotionPolicy();
  const [visible, setVisible] = useState(() => !disabled && !shouldSkipIntro());
  const [opened, setOpened] = useState(false);
  const [leaving, setLeaving] = useState(false);
  useEffect(() => {
    if (!visible || reduced || disabled) return;
    const open = () => setOpened(true);
    const timer = setTimeout(open, 350);
    window.addEventListener('scroll', open, { passive: true, once: true });
    return () => { clearTimeout(timer); window.removeEventListener('scroll', open); };
  }, [visible, reduced, disabled]);
  useEffect(() => {
    if (!opened) return;
    const timer = setTimeout(() => setLeaving(true), 1750);
    return () => clearTimeout(timer);
  }, [opened]);
  useEffect(() => {
    if (!leaving) return;
    const timer = setTimeout(() => setVisible(false), 650);
    return () => clearTimeout(timer);
  }, [leaving]);
  function skip() {
    setVisible(false);
    document.getElementById('hero-cta')?.focus({ preventScroll: true });
  }
  if (!visible || reduced || disabled) return null;
  return <motion.div className="intro-overlay" data-testid="laptop-intro" initial={{ opacity: 1 }} animate={{ opacity: leaving ? 0 : 1 }} transition={{ duration: .6 }} onClick={() => setOpened(true)}>
    <button className="intro-skip button mono" onClick={skip}>Skip intro</button>
    <motion.div className="laptop-scene" aria-hidden="true" animate={{ scale: leaving ? .94 : 1, y: opened ? 0 : [0, -4, 0] }} transition={{ scale: { duration: .6 }, y: { duration: 3, repeat: opened ? 0 : Infinity } }}>
      <motion.div className="laptop-lid" initial={{ rotateX: -90 }} animate={{ rotateX: opened ? 10 : -90 }} transition={{ duration: 1.4, ease: [.16, 1, .3, 1] }}>
        <motion.div className="laptop-screen" inert initial={{ opacity: 0 }} animate={{ opacity: opened ? 1 : 0 }} transition={{ delay: .4, duration: .5 }}><Hero embedded /></motion.div>
      </motion.div>
      <div className="laptop-base"><div className="keyboard-deck" /><div className="trackpad" /></div>
      <div className="seam-glow" />
    </motion.div>
  </motion.div>;
}
