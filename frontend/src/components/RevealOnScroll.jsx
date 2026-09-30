import { motion } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';

export default function RevealOnScroll({ children, delay = 0, calm = false, className = '' }) {
  const reduced = useMotionPolicy();
  return <motion.div className={className} initial={reduced ? false : { opacity: 0, y: calm ? 0 : 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .1 }} transition={{ duration: .55, delay: reduced ? 0 : delay, ease: 'easeOut' }}>{children}</motion.div>;
}
