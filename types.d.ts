export const CONFIG: {
  BASE: string;
  SR: 16000;
  N_FFT: 512;
  HOP: 160;
  WIN: 400;
  N_MELS: 128;
  FMIN: 0;
  FMAX: 8000;
  PREEMPH: 0.97;
  LOG_GUARD: 1e-10;
  NEW_FRAMES: 56;
  CACHE_FRAMES: 9;
  LEFT: 70;
  LAYERS: 24;
  D_MODEL: 1024;
  DEC_HID: 640;
  DEC_LAYERS: 2;
  VOCAB: 13088;
  BLANK: 13087;
  MAX_SYM: 10;
};

export function buildMelFB(): Float32Array[];
export function buildWindow(): Float32Array;
export function computeMelOffline(
  x: Float32Array,
  melFB: Float32Array[],
  win: Float32Array,
): Float32Array[];

export declare class StreamingMel {
  constructor(melFB: Float32Array[], win: Float32Array);
  push(samples: Float32Array): Float32Array[];
}

export interface DetokResult {
  text: string;
  lang: string | null;
}
export function detok(ids: number[], vocab: string[]): DetokResult;

export const Profiles: {
  readonly TURBO: { rightContext: 0; encoder: string; encoderData: string };
  readonly FAST: { rightContext: 1; encoder: string; encoderData: string };
  readonly BALANCED: { rightContext: 3; encoder: string; encoderData: string };
  readonly NORMAL: { rightContext: 6; encoder: string; encoderData: string };
  readonly HIGH: { rightContext: 13; encoder: string; encoderData: string };
};

export const LANG_TO_ID: Record<string, [id: number, name: string]>;
export function langName(code: string): string | null;
export function langId(code: string): number | null;

export interface AsrEngineCallbacks {
  progress?: (label: string, loaded: number, total: number, cached?: boolean) => void;
  status?: (detail: string) => void;
  partial?: (text: string, lang: string | null, progress?: number) => void;
  ep?: (encoder: boolean, ep: string, note?: string) => void;
}

export interface TranscriptionResult {
  text: string;
  lang: string | null;
  tokens: number;
  timing: {
    encoder: number;
    joint: number;
    decoder: number;
    total: number;
  };
}

export interface SessionResult {
  text: string;
  lang: string | null;
  tokens: number;
  timing: {
    encoder: number;
    joint: number;
    decoder: number;
    total: number;
  } | null;
}

export declare class Session {
  constructor(engine: AsrEngine, langId: number);
  feed(samples: Float32Array): Promise<{ text: string; lang: string | null } | null>;
  end(): Promise<SessionResult | null>;
}

export interface BenchmarkProfileResult {
  profile: string;
  rightContext: number;
  latencyLabel: string;
  processingTimeMs: number;
  audioDurationSec: number;
  rtf: number;
  text: string;
  lang: string | null;
  tokens: number;
  timing: {
    encoder: number;
    joint: number;
    decoder: number;
    total: number;
  };
}

export interface BenchmarkOptions {
  profiles?: string[];
  duration?: number;
  langId?: number;
  warmup?: boolean;
  forceAll?: boolean;
  samples?: Float32Array;
}

export interface AsrEngineOptions {
  profile?: "TURBO" | "FAST" | "BALANCED" | "NORMAL" | "HIGH";
  beamWidth?: number;
  ensureCPU?: boolean;
}

export declare class AsrEngine {
  constructor(callbacks?: AsrEngineCallbacks, options?: AsrEngineOptions);
  readonly ready: boolean;
  readonly profile: string;
  readonly encoderEP: string;

  init(): Promise<void>;
  switchProfile(name: string): Promise<void>;
  transcribe(samples: Float32Array, langId: number): Promise<TranscriptionResult>;
  session(langId: number): Session;
  clearCache(): Promise<void>;
  getPerfStats(): Record<string, { ms: number; calls: number; avg: number }>;
  benchmark(options?: BenchmarkOptions): Promise<BenchmarkProfileResult[]>;
}
