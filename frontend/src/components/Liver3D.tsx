import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { MarchingCubes } from 'three/examples/jsm/objects/MarchingCubes.js'
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { LiverFigure } from './Charts'

/* ---------- the liver's shape: rounded parts blended together (signed distances) ---------- */
type V3 = [number, number, number]
const ell = (x: number, y: number, z: number, c: V3, r: V3) => {
  const a = (x - c[0]) / r[0], b = (y - c[1]) / r[1], d = (z - c[2]) / r[2]
  return (Math.sqrt(a * a + b * b + d * d) - 1) * Math.min(r[0], r[1], r[2])
}
const smin = (a: number, b: number, k: number) => { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25 }
const smax = (a: number, b: number, k: number) => -smin(-a, -b, k)
const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const smooth = (a: number, b: number, x: number) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t) }

/** Negative inside the liver. Viewed from the front: the large right lobe on the left of the screen, the left lobe
 *  tapering to a point on the right, the gallbladder hanging under the right lobe. */
export function liverDistance(x: number, y: number, z: number) {
  let d = ell(x, y, z, [-0.3, 0.02, 0], [0.5, 0.45, 0.42])                                // right lobe, domed
  const t = clamp01((x + 0.05) / 0.9)                                                       // 0 at the middle, 1 at the left tip
  d = smin(d, ell(x, y, z, [0.3, 0.14 + 0.05 * t, 0.02], [0.6, 0.27 * (1 - 0.62 * t), 0.33 * (1 - 0.42 * t)]), 0.2)  // left lobe
  d = smin(d, ell(x, y, z, [-0.02, 0.1, 0], [0.36, 0.3, 0.36]), 0.16)                        // the bridge between them
  d = smax(d, -ell(x, y, z, [0.05, -0.62, 0.16], [0.78, 0.34, 0.56]), 0.12)                  // hollowed underside
  d = smax(d, -ell(x, y, z, [0.04, 0.44, 0.26], [0.022, 0.12, 0.26]), 0.08)                   // shallow groove of the falciform ligament
  d = smin(d, ell(x, y, z, [-0.14, -0.34, 0.2], [0.075, 0.15, 0.08]), 0.07)                  // gallbladder
  return d
}

function liverGeometry() {
  const res = 72
  const mc = new MarchingCubes(res, new THREE.MeshBasicMaterial(), false, false, 90000)
  mc.isolation = 0
  const half = res / 2
  for (let k = 0; k < res; k++) for (let j = 0; j < res; j++) for (let i = 0; i < res; i++)
    mc.field[i + j * res + k * res * res] = -liverDistance((i - half) / half, (j - half) / half, (k - half) / half) * 100
  mc.update()
  const n = (mc as unknown as { count: number }).count
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute((mc.geometry.attributes.position.array as Float32Array).slice(0, n * 3), 3))
  const merged = mergeVertices(g, 1e-4)
  merged.computeVertexNormals()
  mc.geometry.dispose()
  return merged
}

/** Branching vessels from the hilum (under the middle) into both lobes, kept inside the liver. */
function vesselCurves() {
  let seed = 11
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5 }
  const curves: { curve: THREE.CatmullRomCurve3; depth: number }[] = []
  const grow = (start: THREE.Vector3, dir: THREE.Vector3, len: number, depth: number) => {
    if (depth > 4) return
    const pts = [start.clone()]
    const p = start.clone(), d = dir.clone()
    for (let i = 0; i < 6; i++) {
      d.add(new THREE.Vector3(rand(), rand(), rand() * 0.6).multiplyScalar(0.45)).normalize()
      const next = p.clone().addScaledVector(d, len / 6)
      if (liverDistance(next.x, next.y, next.z) > -0.05) break
      pts.push(next); p.copy(next)
    }
    if (pts.length < 3) return
    curves.push({ curve: new THREE.CatmullRomCurve3(pts), depth })
    for (let k = 0; k < (depth < 2 ? 3 : 2); k++)
      grow(p, d.clone().add(new THREE.Vector3(rand(), rand(), rand() * 0.5).multiplyScalar(1.6)).normalize(), len * 0.66, depth + 1)
  }
  const hilum = new THREE.Vector3(-0.02, -0.14, 0.08)
  for (const dir of [[-1, 0.25, 0], [-0.55, 0.85, 0.1], [1, 0.3, 0], [0.25, 1, 0.05], [-0.8, -0.2, 0.2]] as V3[])
    grow(hilum, new THREE.Vector3(...dir).normalize(), 0.42, 0)
  return curves
}

/* ---------- colour: blue where there is no fat, through cyan and yellow to orange where fat is likely ---------- */
const STOPS: [number, THREE.Color][] = ([[0, '#163aa8'], [0.3, '#2f7df6'], [0.5, '#2fcfe6'], [0.68, '#f2d94e'], [0.84, '#ff9a2e'], [1, '#ef4a2f']] as [number, string][])
  .map(([v, c]) => [v, new THREE.Color(c)])
export const LEGEND_GRADIENT = `linear-gradient(to top, ${STOPS.map(([v, c]) => `#${c.getHexString()} ${Math.round(v * 100)}%`).join(', ')})`
function colormap(v: number, out: THREE.Color) {
  for (let i = 1; i < STOPS.length; i++) if (v <= STOPS[i][0]) {
    const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i]
    return out.copy(ca).lerp(cb, (v - a) / (b - a))
  }
  return out.copy(STOPS[STOPS.length - 1][1])
}

function noise3(x: number, y: number, z: number) {
  const hash = (i: number, j: number, k: number) => { const h = Math.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453; return h - Math.floor(h) }
  const one = (x: number, y: number, z: number) => {
    const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z)
    const f = (t: number) => t * t * (3 - 2 * t), a = f(x - i), b = f(y - j), c = f(z - k)
    const l = (p: number, q: number, t: number) => p + (q - p) * t
    return l(l(l(hash(i, j, k), hash(i + 1, j, k), a), l(hash(i, j + 1, k), hash(i + 1, j + 1, k), a), b),
             l(l(hash(i, j, k + 1), hash(i + 1, j, k + 1), a), l(hash(i, j + 1, k + 1), hash(i + 1, j + 1, k + 1), a), b), c)
  }
  return 0.6 * one(x * 3, y * 3, z * 3) + 0.4 * one(x * 7 + 5, y * 7, z * 7)
}

// A see-through glowing surface: brighter at the edges (where the surface turns away), and fat colours glow.
const glowVertex = `
  varying vec3 vColor; varying vec3 vNormal; varying vec3 vView;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal); vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`
const glowFragment = `
  uniform float uBase; uniform float uRim;
  varying vec3 vColor; varying vec3 vNormal; varying vec3 vView;
  void main() {
    float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
    float warm = smoothstep(0.35, 0.75, vColor.r);            // yellow, orange and red glow more
    vec3 c = vColor * (uBase + uRim * rim + 0.45 * warm);
    gl_FragColor = vec4(c, clamp(uBase + rim, 0.0, 1.0));
  }`

interface View { fat: number }

/** One or two livers (e.g. today and in 12 months) in a dark panel, turning together. `fat` (0-100) is the chance of
 *  a lot of fat: it sets how much of the liver lights up and how warm the colour is. Changes animate. */
export function Liver3D({ views, label, stacked = false }: { views: View[]; label: string; stacked?: boolean }) {
  const host = useRef<HTMLDivElement>(null)
  const targets = useRef(views.map(v => v.fat))
  targets.current = views.map(v => v.fat)
  const count = views.length
  const [noWebGL, setNoWebGL] = useState(false)

  useEffect(() => {
    const el = host.current
    if (!el) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }) } catch { setNoWebGL(true); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setScissorTest(true)
    el.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50)
    camera.position.set(0, 0.25, 5)
    const shape = liverGeometry()
    const pos = shape.attributes.position as THREE.BufferAttribute
    const pattern = new Float32Array(pos.count)
    for (let i = 0; i < pos.count; i++) pattern[i] = noise3(pos.getX(i), pos.getY(i), pos.getZ(i))
    const vessels = vesselCurves()

    // a soft blue halo behind each liver
    const haloCanvas = document.createElement('canvas'); haloCanvas.width = haloCanvas.height = 128
    const hc = haloCanvas.getContext('2d')!
    const grad = hc.createRadialGradient(64, 64, 0, 64, 64, 64)
    grad.addColorStop(0, 'rgba(60,120,255,0.55)'); grad.addColorStop(1, 'rgba(60,120,255,0)')
    hc.fillStyle = grad; hc.fillRect(0, 0, 128, 128)
    const haloTexture = new THREE.CanvasTexture(haloCanvas)

    const scenes = Array.from({ length: count }, () => {
      const scene = new THREE.Scene()
      const geometry = shape.clone()
      const colors = new Float32Array(pos.count * 3)
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      const surface = new THREE.ShaderMaterial({ vertexShader: glowVertex, fragmentShader: glowFragment, vertexColors: true, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uBase: { value: 0.2 }, uRim: { value: 0.85 } } })
      const mesh = new THREE.ShaderMaterial({ vertexShader: glowVertex, fragmentShader: glowFragment, vertexColors: true, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, wireframe: true, uniforms: { uBase: { value: 0.07 }, uRim: { value: 0.25 } } })
      const group = new THREE.Group()
      group.add(new THREE.Mesh(geometry, surface), new THREE.Mesh(geometry, mesh))
      const vesselMaterial = new THREE.MeshBasicMaterial({ color: '#7fd3ff', transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false })
      for (const { curve, depth } of vessels)
        group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.008 * (1 - depth * 0.15), 6), vesselMaterial))
      group.scale.setScalar(1.4)
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTexture, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.6 }))
      halo.scale.set(4.6, 3.2, 1); halo.position.z = -1.2
      halo.visible = false   // the panel's own background glow is used instead (a halo per view shows a seam)
      scene.add(halo, group)
      return { scene, geometry, colors, shown: -1, painted: -1, materials: [surface, mesh, vesselMaterial] as THREE.Material[] }
    })

    const c = new THREE.Color()
    const paint = (s: typeof scenes[number], f: number) => {   // f: 0-1, the chance of a lot of fat
      const edge = 0.92 - 0.45 * f
      for (let i = 0; i < pos.count; i++) {
        const v = 0.14 + smooth(edge - 0.14, edge + 0.14, pattern[i]) * (0.12 + 0.68 * f)
        colormap(Math.min(v, 0.95), c)
        s.colors[i * 3] = c.r; s.colors[i * 3 + 1] = c.g; s.colors[i * 3 + 2] = c.b
      }
      s.geometry.attributes.color.needsUpdate = true
    }

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true
    controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.6
    controls.addEventListener('start', () => { controls.autoRotate = false })
    controls.saveState()
    const reset = () => { controls.reset(); controls.autoRotate = !reduced }
    renderer.domElement.addEventListener('dblclick', reset)

    let W = 1, H = 1
    const resize = () => {
      W = el.clientWidth; H = el.clientHeight
      renderer.setSize(W, H, false)
      const vw = stacked ? W : W / count, vh = stacked ? H / count : H
      camera.aspect = vw / vh
      camera.position.setLength(5 * Math.max(1, 1.45 / camera.aspect))
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize); ro.observe(el); resize()

    let frame = 0
    const tick = () => {
      controls.update()
      scenes.forEach((s, i) => {
        const goal = targets.current[i] ?? 0
        s.shown = s.shown < 0 ? goal : s.shown + (goal - s.shown) * (reduced ? 1 : 0.08)
        const f = clamp01(s.shown / 100)
        if (Math.abs(f - s.painted) > 0.002) { paint(s, f); s.painted = f }
        const x = stacked ? 0 : (W / count) * i, y = stacked ? (H / count) * (count - 1 - i) : 0
        const vw = stacked ? W : W / count, vh = stacked ? H / count : H
        renderer.setViewport(x, y, vw, vh); renderer.setScissor(x, y, vw, vh)
        renderer.render(s.scene, camera)
      })
      frame = requestAnimationFrame(tick)
    }
    tick()
    return () => {
      cancelAnimationFrame(frame); ro.disconnect(); controls.dispose()
      renderer.domElement.removeEventListener('dblclick', reset)
      scenes.forEach(s => { s.materials.forEach(m => m.dispose()); s.scene.traverse(o => { if (o instanceof THREE.Mesh) o.geometry.dispose() }) })
      shape.dispose(); haloTexture.dispose(); renderer.dispose(); el.removeChild(renderer.domElement)
    }
  }, [count, stacked])

  if (noWebGL) return <div className="liver3d flat" role="img" aria-label={label}>{views.map((v, i) => <LiverFigure key={i} chance={v.fat} />)}</div>
  return <div ref={host} className="liver3d" role="img" aria-label={label} />
}

export default Liver3D
