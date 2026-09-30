import './navigation.css';

export default function Navigation() {
  return <>
    <a className="skip-link button" href="#main">Skip to content</a>
    <header className="site-header page-shell">
      <a className="wordmark" href="#hero" aria-label="Ariadne home"><svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><path d="M5 22V6h18v16H11V12h6v4" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>Ariadne<span className="wordmark-period">.</span></a>
      <nav aria-label="Main navigation"><a href="#pipeline">Pipeline</a><a href="#benchmarks">Evidence</a><a href="#demo">Live demo</a></nav>
      <span className="header-meta mono">PRISM / 2026</span>
    </header>
  </>;
}
