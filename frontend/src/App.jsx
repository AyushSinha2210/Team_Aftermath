import Navigation from './components/Navigation';
import Hero from './components/Hero';
import LaptopReveal from './components/LaptopReveal';
import Problem from './components/Problem';
import Pipeline from './components/Pipeline';
import Benchmarks from './components/Benchmarks';
import Ablation from './components/Ablation';
import Versioning from './components/Versioning';
import Footer from './components/Footer';
import TokenShowcase from './components/TokenShowcase';
import LiveDemo from './components/LiveDemo';

export default function App() {
  if (window.location.pathname === '/design-system') return <TokenShowcase />;
  return <><Navigation /><LaptopReveal /><main className="page-shell" id="main" tabIndex={-1}><Hero /><Problem /><Pipeline /><Benchmarks /><LiveDemo /><Ablation /><Versioning /></main><Footer /></>;
}
