/**
 * APNext H3 — reference API client
 *
 * A dependency-free TypeScript client for driving the exported H3 workflows
 * (the *.api.json files next to this file) against a running ComfyUI server:
 * upload the song and reference images, patch the inputs your front-end
 * exposes, queue, stream progress over the websocket, answer the Dailies Gate
 * (approve / punch-up / new take / undo / cut), and collect the saved clips.
 *
 * Runs in any browser or Node >= 22 (global fetch + WebSocket). In a SvelteKit
 * app, call this from server routes (BFF) and fan events out to your UI.
 *
 *   const client = new H3Client("http://127.0.0.1:8188");
 *   const wf = structuredClone(minimalWorkflow);        // *.api.json content
 *   const song = await client.upload(songFile);
 *   H3Client.patchByClass(wf, "LoadAudio", { audio: song });
 *   H3Client.patchByClass(wf, "LoadImage", { image: await client.upload(photo) });
 *   H3Client.patchByClass(wf, "H3MusicVideoMinimal", {
 *     lyrics: "[0:12] I walked the wire in the rain",
 *     performance: 80, pace: 30, wildness: 45,
 *     prompt_mode: "Ref2VA (bind reference images)",
 *   });
 *   const run = await client.run(wf, {
 *     onProgress: (e) => console.log(e.type),
 *     onGate: async (gate) => gate.approve(),           // or punchUp("notes")
 *   });
 *   console.log(run.outputs);                            // saved clip URLs
 */

export type ApiWorkflow = Record<
  string,
  { class_type: string; inputs: Record<string, unknown> }
>;

export interface GateEvent {
  token: string;
  node: string;
  text: string;
  count: number;
  can_reroll: boolean;
  can_undo: boolean;
  revision: number;
  status: string;
  /** Continue rendering exactly `text` (edit it first if you like). */
  approve(text?: string): Promise<void>;
  /** Rewrite `takes` (e.g. "2,4-5"; empty = all) with the director's notes. */
  punchUp(feedback: string, takes?: string, text?: string): Promise<void>;
  /** A noticeably different version of the selected takes. */
  newTake(takes?: string): Promise<void>;
  undo(): Promise<void>;
  cut(): Promise<void>;
}

export interface RunResult {
  promptId: string;
  /** Every saved output as a downloadable URL, in completion order. */
  outputs: string[];
  history: unknown;
}

export class H3Client {
  constructor(
    private base = "http://127.0.0.1:8188",
    private clientId = crypto.randomUUID(),
  ) {}

  /** Upload a song / image into ComfyUI's input folder; returns the filename
   *  to patch into a LoadAudio / LoadImage node. */
  async upload(file: Blob & { name?: string }, name?: string): Promise<string> {
    const form = new FormData();
    form.append("image", file, name ?? file.name ?? "upload.bin");
    const r = await fetch(`${this.base}/upload/image`, { method: "POST", body: form });
    if (!r.ok) throw new Error(`upload failed: ${r.status} ${await r.text()}`);
    const data = (await r.json()) as { name: string; subfolder?: string };
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
  }

  /** Merge `patch` into every node of `classType` (usually exactly one). */
  static patchByClass(wf: ApiWorkflow, classType: string, patch: Record<string, unknown>) {
    let hits = 0;
    for (const node of Object.values(wf)) {
      if (node.class_type === classType) {
        Object.assign(node.inputs, patch);
        hits++;
      }
    }
    if (!hits) throw new Error(`no node of class ${classType} in workflow`);
    return hits;
  }

  /** Queue a workflow and resolve when it finishes (or the gate is answered
   *  for as many rounds as it takes). */
  async run(
    wf: ApiWorkflow,
    handlers: {
      onProgress?: (event: { type: string; data: unknown }) => void;
      onGate?: (gate: GateEvent) => void | Promise<void>;
    } = {},
  ): Promise<RunResult> {
    const wsUrl =
      this.base.replace(/^http/, "ws") + `/ws?clientId=${encodeURIComponent(this.clientId)}`;
    const ws = new WebSocket(wsUrl);
    await new Promise<void>((ok, err) => {
      ws.addEventListener("open", () => ok());
      ws.addEventListener("error", () => err(new Error(`websocket failed: ${wsUrl}`)));
    });

    const queued = await fetch(`${this.base}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: wf, client_id: this.clientId }),
    });
    if (!queued.ok) throw new Error(`queue failed: ${queued.status} ${await queued.text()}`);
    const { prompt_id: promptId } = (await queued.json()) as { prompt_id: string };

    await new Promise<void>((done, fail) => {
      ws.addEventListener("message", async (msg) => {
        if (typeof msg.data !== "string") return; // binary previews
        const event = JSON.parse(msg.data) as { type: string; data: any };
        handlers.onProgress?.(event);

        if (event.type === "apnext.h3.review_gate" && handlers.onGate) {
          const d = event.data;
          const decide = (body: Record<string, unknown>) =>
            fetch(`${this.base}/apnext/h3/review_gate`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ token: d.token, text: d.text, ...body }),
            }).then((r) => {
              if (!r.ok) throw new Error(`gate decision failed: ${r.status}`);
            });
          await handlers.onGate({
            ...d,
            approve: (text?: string) =>
              decide({ action: "approve", ...(text != null ? { text } : {}) }),
            punchUp: (feedback: string, takes = "", text?: string) =>
              decide({ action: "reroll", feedback, scenes: takes, ...(text != null ? { text } : {}) }),
            newTake: (takes = "") => decide({ action: "reroll", variant: true, scenes: takes }),
            undo: () => decide({ action: "undo" }),
            cut: () => decide({ action: "stop" }),
          });
        }

        if (event.type === "execution_success" && event.data?.prompt_id === promptId) done();
        if (
          (event.type === "execution_error" || event.type === "execution_interrupted") &&
          event.data?.prompt_id === promptId
        ) {
          fail(new Error(`${event.type}: ${JSON.stringify(event.data)}`));
        }
      });
    }).finally(() => ws.close());

    const historyResponse = await fetch(`${this.base}/history/${promptId}`);
    const history = ((await historyResponse.json()) as any)[promptId];
    const outputs: string[] = [];
    for (const nodeOutput of Object.values<any>(history?.outputs ?? {})) {
      for (const kind of ["images", "video", "videos", "audio", "gifs"]) {
        for (const f of nodeOutput?.[kind] ?? []) {
          const q = new URLSearchParams({
            filename: f.filename,
            subfolder: f.subfolder ?? "",
            type: f.type ?? "output",
          });
          outputs.push(`${this.base}/view?${q}`);
        }
      }
    }
    return { promptId, outputs, history };
  }
}
