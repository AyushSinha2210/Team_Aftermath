import { Component } from 'react';

export default class ErrorBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <main className="page-shell section"><h1>Ariadne</h1><p className="section-note" role="alert">The interface could not load. Reload to try again.</p><button className="button button-primary mt-6" onClick={() => window.location.reload()}>Reload page</button></main>;
    return this.props.children;
  }
}
