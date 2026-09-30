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

export default function App() {
  if (window.location.pathname === '/design-system') return <TokenShowcase />;
  return <><Navigation /><LaptopReveal /><main className="page-shell" id="main" tabIndex={-1}><Hero /><Problem /><Pipeline /><Benchmarks /><section className="section" id="demo"><h2>Try it yourself</h2><p className="section-note">The search interface is being connected.</p></section><Ablation /><Versioning /></main><Footer /></>;
}
