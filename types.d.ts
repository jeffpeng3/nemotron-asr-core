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
  readonly TURBO: { encoder: string; encoderData: string };
  readonly FAST: { encoder: string; encoderData: string };
  readonly BALANCED: { encoder: string; encoderData: string };
  readonly NORMAL: { encoder: string; encoderData: string };
  readonly HIGH: { encoder: string; encoderData: string };
};

export const LANG_TO_ID: Record<string, [id: number, name: string]>;
export function langName(code: string): string | null;
export function langId(code: string): number | null;

export interface VadOptions {
  threshold?: number;
  minSpeech?: number;
  minSilence?: number;
  hold?: number;
  sr?: number;
}

export interface AsrEngineCallbacks {
  progress?: (label: string, loaded: number, total: number, cached?: boolean) => void;
  status?: (detail: string) => void;
  partial?: (text: string, lang: string | null, progress?: number) => void;
  ep?: (encoder: boolean, ep: string, note?: string) => void;
  speechStart?: () => void;
  speechEnd?: () => void;
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

export declare class EnergyVAD {
  static computeRMS(samples: Float32Array): number;
  constructor(opts?: VadOptions);
  readonly active: boolean;
  process(level: number, nSamples: number): boolean;
}

export declare class Session {
  constructor(engine: AsrEngine, langId: number, vadOptions?: boolean | VadOptions);
  readonly speaking: boolean;
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
  vad?: boolean | VadOptions;
}

export declare class AsrEngine {
  constructor(callbacks?: AsrEngineCallbacks, options?: AsrEngineOptions);
  readonly ready: boolean;
  readonly profile: string;
  readonly encoderEP: string;

  init(): Promise<void>;
  switchProfile(name: string): Promise<void>;
  transcribe(samples: Float32Array, langId: number): Promise<TranscriptionResult>;
  session(langId: number, vadOverride?: boolean | VadOptions): Session;
  clearCache(): Promise<void>;
  getPerfStats(): Record<string, { ms: number; calls: number; avg: number }>;
  benchmark(options?: BenchmarkOptions): Promise<BenchmarkProfileResult[]>;
}
