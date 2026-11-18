"use client";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type V = [number, number, number];
export interface Segments { shoulder_width: number; hip_width: number; upper_arm: number; forearm: number; thigh: number; shin: number; torso: number }
/** Typical segment lengths as a fraction of stature (Drillis & Contini averages) - the grey reference figure. */
export const AVERAGE = (h: number): Segments => ({ shoulder_width: 0.259 * h, hip_width: 0.191 * h, upper_arm: 0.186 * h, forearm: 0.146 * h, thigh: 0.245 * h, shin: 0.246 * h, torso: 0.30 * h });

/** T-pose joint positions from segment lengths (cm). Origin = pelvis centre; y up. */
export function pose(s: Segments): Record<string, V> {
  const sh = s.torso, sw = s.shoulder_width / 2, hw = s.hip_width / 2, p: Record<string, V> = {};
  p.pelvis = [0, 0, 0]; p.neck = [0, sh, 0]; p.head = [0, sh + 0.13 * (s.torso / 0.3), 0];
  for (const [side, d] of [["l", -1], ["r", 1]] as const) {
    p[`${side}sh`] = [d * sw, sh, 0]; p[`${side}el`] = [d * (sw + s.upper_arm), sh, 0]; p[`${side}wr`] = [d * (sw + s.upper_arm + s.forearm), sh, 0];
    p[`${side}hip`] = [d * hw, 0, 0]; p[`${side}kn`] = [d * hw, -s.thigh, 0]; p[`${side}an`] = [d * hw, -s.thigh - s.shin, 0];
  }
  return p;
}
const BONES: [string, string][] = [["pelvis", "neck"], ["neck", "head"], ["lsh", "rsh"], ["lsh", "lel"], ["lel", "lwr"], ["rsh", "rel"], ["rel", "rwr"], ["lhip", "rhip"], ["lhip", "lkn"], ["lkn", "lan"], ["rhip", "rkn"], ["rkn", "ran"]];

function Bone({ a, b, color, opacity }: { a: V; b: V; color: string; opacity: number }) {
  const { pos, quat, len } = useMemo(() => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b), d = vb.clone().sub(va);
    return { pos: va.clone().add(vb).multiplyScalar(0.5), len: d.length(), quat: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize()) };
  }, [a, b]);
  return <mesh position={pos} quaternion={quat}><cylinderGeometry args={[1.1, 1.1, len, 10]} /><meshStandardMaterial color={color} transparent opacity={opacity} /></mesh>;
}

function Figure({ seg, color, opacity }: { seg: Segments; color: string; opacity: number }) {
  const p = useMemo(() => pose(seg), [seg]);
  return (<group>{BONES.map(([a, b]) => <Bone key={a + b} a={p[a]} b={p[b]} color={color} opacity={opacity} />)}
    {Object.entries(p).map(([k, v]) => <mesh key={k} position={v}><sphereGeometry args={[k === "head" ? 4.5 : 2, 14, 14]} /><meshStandardMaterial color={color} transparent opacity={opacity} /></mesh>)}</group>);
}

function Spin({ children }: { children: React.ReactNode }) {
  const g = useRef<THREE.Group>(null);
  useFrame((_, dt) => { if (g.current) g.current.rotation.y += dt * 0.5; });
  return <group ref={g}>{children}</group>;
}

function webglOk(): boolean { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch { return false; } }

/** Frame the figure from its real extent: vertical centre -> group offset, camera distance from the larger of the two ranges. */
export function framing(seg: Segments, avg: Segments, fov = 40, aspect = 1.55) {
  const ys: number[] = [], xs: number[] = [];
  for (const sg of [seg, avg]) for (const v of Object.values(pose(sg))) { ys.push(v[1]); xs.push(v[0]); }
  const top = Math.max(...ys) + 6, bottom = Math.min(...ys) - 3, half = Math.max(...xs.map(Math.abs)) + 6;
  const need = Math.max(top - bottom, (2 * half) / aspect) * 1.18;
  return { offsetY: -(top + bottom) / 2, distance: need / (2 * Math.tan((fov * Math.PI) / 360)) };
}

export function SkeletonViewer3D({ segments, heightCm }: { segments: Segments; heightCm: number }) {
  const f = framing(segments, AVERAGE(heightCm));
  if (typeof document !== "undefined" && !webglOk()) return <p className="text-sm text-mute">3D preview needs WebGL, which isn&apos;t available here. Your measurements are listed below.</p>;
  return (
    <div className="h-72 w-full overflow-hidden rounded-2xl border border-line bg-raise" role="img" aria-label="3D view of your body proportions compared with average proportions" data-testid="skeleton-3d">
      <Canvas camera={{ position: [0, 0, f.distance], fov: 40 }} dpr={[1, 2]}>
        <ambientLight intensity={0.8} /><directionalLight position={[50, 80, 100]} intensity={1.2} />
        <group position={[0, f.offsetY, 0]}><Spin>
          <Figure seg={AVERAGE(heightCm)} color="#93a3b2" opacity={0.25} /><Figure seg={segments} color="#4a74ff" opacity={1} /></Spin></group>
      </Canvas>
    </div>);
}
