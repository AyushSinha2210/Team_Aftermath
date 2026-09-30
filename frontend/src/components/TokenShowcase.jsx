import { Button, QueryInput } from './Controls';

const names = ['bg', 'surface', 'surface-raised', 'border', 'gold', 'gold-dim', 'teal', 'teal-dim', 'text-primary', 'text-secondary', 'text-muted', 'success', 'error', 'focus-ring'];
export default function TokenShowcase() {
  return <main className="page-shell section space-y-8">
    <h1 className="text-6xl">Ariadne design system</h1>
    <p>Fraunces for headlines. Manrope for the interface. <span className="mono">JetBrains Mono 0.739</span> for precision.</p>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">{names.map(name => <div key={name}><div className="h-16 rounded border border-border" style={{ background: `var(--${name})` }} /><p className="mono text-sm mt-3">{name}</p></div>)}</div>
    <div className="flex gap-4 flex-wrap"><Button>Default and hover</Button><Button className="button-primary">Primary</Button><Button disabled>Disabled</Button><Button autoFocus>Keyboard focus</Button></div>
    <QueryInput aria-label="Default input" placeholder="Default and focus" />
    <QueryInput aria-label="Invalid input" error aria-describedby="showcase-error" defaultValue="" placeholder="Error state" />
    <p id="showcase-error" className="inline-error">Please enter a query.</p>
    <a href="/">Return to Ariadne</a>
  </main>;
}
