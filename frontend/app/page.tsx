"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Umbral de derivacion a revision humana.
 *
 * Procede del analisis de calibracion del modelo sobre las 900 imagenes de
 * prueba reservadas: ninguno de los 10 errores cometidos supero una confianza
 * de 0,99, de modo que este umbral habria capturado la totalidad de los fallos
 * revisando solo el 5,8% de los casos.
 */
const UMBRAL_REVISION = 0.99;

type Respuesta = {
  prediction: { label: string; confidence: number; class_id: number };
  explainability: {
    heatmap_base64: string;
    overlay_base64: string;
    description: string;
  };
  performance: {
    preprocess_time_ms: number;
    inference_time_ms: number;
    explainability_time_ms: number;
    total_latency_ms: number;
    model_used: string;
  };
};

const FORMATOS = ["image/jpeg", "image/png", "image/webp"];
const TAMANO_MAXIMO = 8 * 1024 * 1024;

// Radiografias incluidas para poder demostrar el sistema sin depender de
// tener un archivo a mano. Ninguna pertenece al conjunto de entrenamiento.
const EJEMPLOS = [
  { etiqueta: "Ejemplo con hallazgo", archivo: "/ejemplos/ejemplo-anomalia.jpeg" },
  { etiqueta: "Ejemplo sin hallazgos", archivo: "/ejemplos/ejemplo-normal.jpeg" },
];

export default function Pagina() {
  const [imagen, setImagen] = useState<string | null>(null);
  const [nombreArchivo, setNombreArchivo] = useState("");
  const [resultado, setResultado] = useState<Respuesta | null>(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vista, setVista] = useState<"overlay" | "heatmap">("overlay");
  const [arrastrando, setArrastrando] = useState(false);
  const entradaRef = useRef<HTMLInputElement>(null);

  // Se despierta el contenedor al cargar la pagina: se apaga tras un minuto
  // sin trafico, de modo que el primer analisis cargaria con el arranque en
  // frio si no se anticipa aqui.
  useEffect(() => {
    fetch("/api/despertar").catch(() => {});
  }, []);

  const cargarArchivo = useCallback((archivo: File) => {
    setError(null);
    setResultado(null);

    if (!FORMATOS.includes(archivo.type)) {
      setError("Formato no admitido. Usa una imagen JPEG, PNG o WebP.");
      return;
    }
    if (archivo.size > TAMANO_MAXIMO) {
      setError(
        `La imagen pesa ${(archivo.size / 1024 / 1024).toFixed(1)} MB y el maximo es 8 MB.`,
      );
      return;
    }

    const lector = new FileReader();
    lector.onload = () => {
      setImagen(lector.result as string);
      setNombreArchivo(archivo.name);
    };
    lector.onerror = () => setError("No se pudo leer el archivo.");
    lector.readAsDataURL(archivo);
  }, []);

  async function analizar() {
    if (!imagen) return;
    setCargando(true);
    setError(null);
    setResultado(null);

    try {
      const respuesta = await fetch("/api/clasificar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imagen.split(",")[1] }),
      });
      const datos = await respuesta.json();

      if (!respuesta.ok) {
        setError(datos.detail ?? `Error ${respuesta.status} al analizar la imagen.`);
        return;
      }
      setResultado(datos as Respuesta);
    } catch {
      setError("No se pudo conectar con el servicio. Revisa tu conexion.");
    } finally {
      setCargando(false);
    }
  }

  async function cargarEjemplo(ruta: string, etiqueta: string) {
    setError(null);
    setResultado(null);
    try {
      const respuesta = await fetch(ruta);
      const blob = await respuesta.blob();
      const lector = new FileReader();
      lector.onload = () => {
        setImagen(lector.result as string);
        setNombreArchivo(etiqueta);
      };
      lector.readAsDataURL(blob);
    } catch {
      setError("No se pudo cargar la radiografía de ejemplo.");
    }
  }

  function limpiar() {
    setImagen(null);
    setNombreArchivo("");
    setResultado(null);
    setError(null);
    if (entradaRef.current) entradaRef.current.value = "";
  }

  const esAnomalia = resultado?.prediction.label === "Anomaly";
  const requiereRevision =
    resultado !== null && resultado.prediction.confidence < UMBRAL_REVISION;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto max-w-5xl px-5 py-10">
        <header className="mb-8 border-b border-slate-200 pb-6 dark:border-slate-800">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Clasificación binaria radiológica
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            Sistema de apoyo al diagnóstico basado en redes neuronales convolucionales.
            Analiza radiografías de tórax y señala las regiones que fundamentan su
            decisión mediante Grad-CAM.
          </p>
        </header>

        <section
          onDragOver={(e) => {
            e.preventDefault();
            setArrastrando(true);
          }}
          onDragLeave={() => setArrastrando(false)}
          onDrop={(e) => {
            e.preventDefault();
            setArrastrando(false);
            const archivo = e.dataTransfer.files?.[0];
            if (archivo) cargarArchivo(archivo);
          }}
          className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
            arrastrando
              ? "border-sky-500 bg-sky-50 dark:bg-sky-950/30"
              : "border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900"
          }`}
        >
          <input
            ref={entradaRef}
            type="file"
            accept={FORMATOS.join(",")}
            className="hidden"
            onChange={(e) => {
              const archivo = e.target.files?.[0];
              if (archivo) cargarArchivo(archivo);
            }}
          />

          {imagen ? (
            <div className="space-y-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imagen}
                alt="Radiografía seleccionada"
                className="mx-auto max-h-72 rounded-lg object-contain"
              />
              <p className="text-xs text-slate-500 dark:text-slate-400">{nombreArchivo}</p>
              <div className="flex flex-wrap justify-center gap-3">
                <button
                  onClick={analizar}
                  disabled={cargando}
                  className="rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {cargando ? "Analizando…" : "Analizar radiografía"}
                </button>
                <button
                  onClick={limpiar}
                  disabled={cargando}
                  className="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-medium transition hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
                >
                  Elegir otra
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Arrastra una radiografía aquí o selecciónala desde tu equipo
              </p>
              <button
                onClick={() => entradaRef.current?.click()}
                className="rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-sky-700"
              >
                Seleccionar imagen
              </button>
              <p className="text-xs text-slate-400">JPEG, PNG o WebP · hasta 8 MB</p>

              <div className="pt-2">
                <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
                  ¿No tienes una a mano? Prueba con estas:
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {EJEMPLOS.map(({ etiqueta, archivo }) => (
                    <button
                      key={archivo}
                      onClick={() => cargarEjemplo(archivo, etiqueta)}
                      className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium transition hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                    >
                      {etiqueta}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        {cargando && (
          <p className="mt-4 text-center text-sm text-slate-500 dark:text-slate-400">
            Si es el primer análisis tras un rato de inactividad, iniciar el servicio
            puede tardar unos segundos.
          </p>
        )}

        {error && (
          <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
            {error}
          </div>
        )}

        {resultado && (
          <section className="mt-8 space-y-6">
            <div
              className={`rounded-xl border p-6 ${
                esAnomalia
                  ? "border-amber-300 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/30"
                  : "border-emerald-300 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/30"
              }`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Resultado
                  </p>
                  <p className="mt-1 text-2xl font-semibold">
                    {esAnomalia ? "Hallazgo anómalo" : "Sin hallazgos"}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    Confianza
                  </p>
                  <p className="mt-1 text-2xl font-semibold tabular-nums">
                    {(resultado.prediction.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {requiereRevision && (
                <p className="mt-4 rounded-lg bg-white/70 p-3 text-sm dark:bg-black/20">
                  <strong className="font-medium">Se recomienda revisión humana.</strong>{" "}
                  La confianza está por debajo del umbral de {UMBRAL_REVISION}, valor que
                  en la evaluación del modelo delimitó la totalidad de sus errores.
                </p>
              )}
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold">Regiones de atención</h2>
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                    En rojo, las zonas que más pesaron en la decisión del modelo
                  </p>
                </div>
                <div className="flex gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-800">
                  {(["overlay", "heatmap"] as const).map((modo) => (
                    <button
                      key={modo}
                      onClick={() => setVista(modo)}
                      className={`rounded px-3 py-1.5 text-xs font-medium transition ${
                        vista === modo
                          ? "bg-white shadow-sm dark:bg-slate-700"
                          : "text-slate-600 dark:text-slate-400"
                      }`}
                    >
                      {modo === "overlay" ? "Superpuesto" : "Solo mapa"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <figure>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imagen ?? ""}
                    alt="Radiografía original"
                    className="w-full rounded-lg border border-slate-200 object-contain dark:border-slate-700"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500 dark:text-slate-400">
                    Original
                  </figcaption>
                </figure>
                <figure>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/jpeg;base64,${
                      vista === "overlay"
                        ? resultado.explainability.overlay_base64
                        : resultado.explainability.heatmap_base64
                    }`}
                    alt="Mapa de atención Grad-CAM"
                    className="w-full rounded-lg border border-slate-200 object-contain dark:border-slate-700"
                  />
                  <figcaption className="mt-2 text-center text-xs text-slate-500 dark:text-slate-400">
                    Grad-CAM
                  </figcaption>
                </figure>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-900">
              <h2 className="mb-4 text-base font-semibold">Rendimiento</h2>
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {(
                  [
                    ["Preprocesamiento", resultado.performance.preprocess_time_ms],
                    ["Inferencia", resultado.performance.inference_time_ms],
                    ["Grad-CAM", resultado.performance.explainability_time_ms],
                    ["Total", resultado.performance.total_latency_ms],
                  ] as const
                ).map(([etiqueta, valor]) => (
                  <div key={etiqueta}>
                    <dt className="text-xs text-slate-500 dark:text-slate-400">
                      {etiqueta}
                    </dt>
                    <dd className="mt-1 text-lg font-semibold tabular-nums">
                      {valor.toFixed(0)}
                      <span className="ml-1 text-xs font-normal text-slate-500">ms</span>
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
                Modelo: {resultado.performance.model_used}
              </p>
            </div>
          </section>
        )}

        <footer className="mt-12 border-t border-slate-200 pt-6 dark:border-slate-800">
          <p className="rounded-lg bg-slate-100 p-4 text-xs leading-relaxed text-slate-600 dark:bg-slate-900 dark:text-slate-400">
            <strong className="font-medium text-slate-800 dark:text-slate-200">
              Aviso.
            </strong>{" "}
            Herramienta desarrollada con fines académicos como trabajo de grado. No
            constituye un dispositivo médico ni sustituye el juicio de un profesional de
            la salud. Sus resultados no deben emplearse para tomar decisiones clínicas.
          </p>
        </footer>
      </div>
    </main>
  );
}
