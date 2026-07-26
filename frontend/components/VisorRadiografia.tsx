"use client";

import { useCallback, useLayoutEffect, useRef, useState } from "react";

/**
 * Visor de radiografia con comparacion y ampliacion.
 *
 * El zoom se limita a la resolucion nativa de la imagen: ampliar mas alla de
 * 1:1 solo interpolaria pixeles y mostraria detalle que no existe, lo que en
 * un contexto diagnostico induce a error. El maximo se calcula a partir de las
 * dimensiones reales del archivo, de modo que cada imagen tiene el suyo.
 *
 * El divisor de comparacion recorta en coordenadas de pantalla, no de la
 * imagen: asi la linea permanece fija aunque se desplace o amplie el
 * contenido, que es el comportamiento que espera quien la usa.
 *
 * El mapa de calor NO se despliega por defecto cuando no hay hallazgos. El
 * rojo sobre una radiografia comunica "aqui hay algo" de forma inmediata,
 * aunque el texto diga lo contrario, y la imagen gana a la etiqueta. En un
 * caso normal el mapa senala donde se sustenta la AUSENCIA de hallazgos, que
 * es una lectura distinta y menos intuitiva, de modo que se muestra solo a
 * peticion y acompanado de esa aclaracion.
 */

type Props = {
  original: string;
  superpuesta: string;
  mapa: string;
  esAnomalia: boolean;
};

type Punto = { x: number; y: number };

const PASO_RUEDA = 1.15;

export default function VisorRadiografia({
  original,
  superpuesta,
  mapa,
  esAnomalia,
}: Props) {
  const contenedorRef = useRef<HTMLDivElement>(null);

  // Con hallazgo, la explicabilidad es lo primero que interesa ver. Sin
  // hallazgos, se ofrece pero no se impone.
  const [explicacion, setExplicacion] = useState(esAnomalia);

  const [natural, setNatural] = useState({ ancho: 0, alto: 0 });
  const [caja, setCaja] = useState({ ancho: 0, alto: 0 });
  const [zoom, setZoom] = useState(1);
  const [desplazamiento, setDesplazamiento] = useState<Punto>({ x: 0, y: 0 });
  const [division, setDivision] = useState(50);
  const [capa, setCapa] = useState<"superpuesta" | "mapa">("superpuesta");

  const arrastrando = useRef<"divisor" | "panoramica" | null>(null);
  const ultimoPunto = useRef<Punto>({ x: 0, y: 0 });

  // Escala a la que la imagen encaja en el contenedor. Nunca supera 1: si el
  // archivo es mas pequeno que el area disponible no se agranda, porque eso
  // seria inventar resolucion.
  const escalaAjuste =
    natural.ancho && caja.ancho
      ? Math.min(caja.ancho / natural.ancho, caja.alto / natural.alto, 1)
      : 1;

  const zoomMaximo = escalaAjuste > 0 ? 1 / escalaAjuste : 1;
  const baseAncho = natural.ancho * escalaAjuste;
  const baseAlto = natural.alto * escalaAjuste;
  const enResolucionNativa = zoom >= zoomMaximo - 0.001;

  useLayoutEffect(() => {
    const nodo = contenedorRef.current;
    if (!nodo) return;
    const medir = () =>
      setCaja({ ancho: nodo.clientWidth, alto: nodo.clientHeight });
    medir();
    const observador = new ResizeObserver(medir);
    observador.observe(nodo);
    return () => observador.disconnect();
  }, []);

  // El encuadre no se reinicia con un efecto: quien usa este componente le
  // pasa una `key` distinta en cada analisis, de modo que React lo remonta y
  // todo el estado vuelve a su valor inicial. Reiniciarlo con setState dentro
  // de un efecto provocaria renderizados en cascada.

  const limitar = useCallback(
    (p: Punto, z: number): Punto => {
      const excesoX = Math.max(0, (baseAncho * z - caja.ancho) / 2);
      const excesoY = Math.max(0, (baseAlto * z - caja.alto) / 2);
      return {
        x: Math.min(excesoX, Math.max(-excesoX, p.x)),
        y: Math.min(excesoY, Math.max(-excesoY, p.y)),
      };
    },
    [baseAncho, baseAlto, caja],
  );

  const aplicarZoom = useCallback(
    (nuevoZoom: number, focoX?: number, focoY?: number) => {
      const z = Math.min(zoomMaximo, Math.max(1, nuevoZoom));
      const nodo = contenedorRef.current;
      if (!nodo) return;

      // Se conserva bajo el cursor el mismo punto de la imagen.
      const rect = nodo.getBoundingClientRect();
      const cx = focoX !== undefined ? focoX - rect.left - rect.width / 2 : 0;
      const cy = focoY !== undefined ? focoY - rect.top - rect.height / 2 : 0;

      setDesplazamiento((previo) => {
        const factor = z / zoom;
        return limitar(
          { x: cx - (cx - previo.x) * factor, y: cy - (cy - previo.y) * factor },
          z,
        );
      });
      setZoom(z);
    },
    [zoom, zoomMaximo, limitar],
  );

  function alGirarRueda(e: React.WheelEvent) {
    if (zoomMaximo <= 1) return;
    e.preventDefault();
    aplicarZoom(e.deltaY < 0 ? zoom * PASO_RUEDA : zoom / PASO_RUEDA, e.clientX, e.clientY);
  }

  function moverDivisorDesde(clientX: number) {
    const nodo = contenedorRef.current;
    if (!nodo) return;
    const rect = nodo.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setDivision(Math.min(100, Math.max(0, pct)));
  }

  function alPresionar(e: React.PointerEvent, modo: "divisor" | "panoramica") {
    if (modo === "panoramica" && zoom <= 1) return;
    arrastrando.current = modo;
    ultimoPunto.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (modo === "divisor") moverDivisorDesde(e.clientX);
  }

  function alMover(e: React.PointerEvent) {
    if (!arrastrando.current) return;
    if (arrastrando.current === "divisor") {
      moverDivisorDesde(e.clientX);
      return;
    }
    const dx = e.clientX - ultimoPunto.current.x;
    const dy = e.clientY - ultimoPunto.current.y;
    ultimoPunto.current = { x: e.clientX, y: e.clientY };
    setDesplazamiento((previo) => limitar({ x: previo.x + dx, y: previo.y + dy }, zoom));
  }

  function alSoltar(e: React.PointerEvent) {
    arrastrando.current = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* el puntero pudo liberarse solo */
    }
  }

  function restablecer() {
    setZoom(1);
    setDesplazamiento({ x: 0, y: 0 });
  }

  const transformacion = {
    width: baseAncho || undefined,
    height: baseAlto || undefined,
    transform: `translate(calc(-50% + ${desplazamiento.x}px), calc(-50% + ${desplazamiento.y}px)) scale(${zoom})`,
  } as const;

  const claseImagen = "pointer-events-none absolute left-1/2 top-1/2 max-w-none select-none";

  return (
    <figure className="overflow-hidden rounded-xl border border-[var(--borde)] bg-[var(--panel)]">
      {/* Barra de herramientas */}
      <div className="sin-imprimir flex flex-wrap items-center justify-between gap-3 border-b border-[var(--borde)] px-4 py-2.5">
        {explicacion ? (
          <div className="flex items-center gap-2">
            <div className="flex gap-1 rounded-lg bg-[var(--panel-alto)] p-0.5">
              {(
                [
                  ["superpuesta", "Superpuesto"],
                  ["mapa", "Solo mapa"],
                ] as const
              ).map(([valor, texto]) => (
                <button
                  key={valor}
                  onClick={() => setCapa(valor)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                    capa === valor
                      ? "bg-[var(--acento)] text-white"
                      : "text-[var(--tinta-media)] hover:text-[var(--tinta)]"
                  }`}
                >
                  {texto}
                </button>
              ))}
            </div>
            {!esAnomalia && (
              <button
                onClick={() => setExplicacion(false)}
                className="rounded-md px-2 py-1 text-xs text-[var(--tinta-tenue)] transition hover:text-[var(--tinta)]"
              >
                Ocultar
              </button>
            )}
          </div>
        ) : (
          <button
            onClick={() => setExplicacion(true)}
            className="rounded-lg border border-[var(--borde-fuerte)] px-3 py-1.5 text-xs font-medium text-[var(--tinta-media)] transition hover:border-[var(--acento)]/50 hover:text-[var(--tinta)]"
          >
            Ver regiones que sustentan la decisión
          </button>
        )}

        <div className="flex items-center gap-1.5">
          <span
            className={`mr-1 font-mono text-xs tabular-nums ${
              enResolucionNativa ? "text-[var(--acento-vivo)]" : "text-[var(--tinta-tenue)]"
            }`}
            title={
              enResolucionNativa
                ? "Resolución nativa: un píxel de pantalla por píxel de la imagen"
                : "Ampliación actual"
            }
          >
            {enResolucionNativa ? "1:1" : `${Math.round(zoom * 100)}%`}
          </span>

          <BotonHerramienta
            onClick={() => aplicarZoom(zoom / PASO_RUEDA)}
            desactivado={zoom <= 1}
            titulo="Reducir"
          >
            −
          </BotonHerramienta>
          <BotonHerramienta
            onClick={() => aplicarZoom(zoom * PASO_RUEDA)}
            desactivado={enResolucionNativa}
            titulo="Ampliar"
          >
            +
          </BotonHerramienta>
          <BotonHerramienta
            onClick={() => aplicarZoom(zoomMaximo)}
            desactivado={enResolucionNativa}
            titulo="Ver a resolución nativa"
          >
            <span className="font-mono text-[10px]">1:1</span>
          </BotonHerramienta>
          <BotonHerramienta onClick={restablecer} desactivado={zoom <= 1} titulo="Ajustar">
            <span className="text-[10px]">Ajustar</span>
          </BotonHerramienta>
        </div>
      </div>

      {/* Lienzo */}
      <div
        ref={contenedorRef}
        onWheel={alGirarRueda}
        onPointerDown={(e) => alPresionar(e, "panoramica")}
        onPointerMove={alMover}
        onPointerUp={alSoltar}
        onPointerCancel={alSoltar}
        className="relative h-[clamp(20rem,52vh,34rem)] touch-none overflow-hidden bg-[var(--lienzo)]"
        style={{ cursor: zoom > 1 ? "grab" : "default" }}
      >
        {/* Capa base: radiografia original.
            Se usa <img> y no <Image>: las fuentes son URLs de datos generadas
            en el navegador, que el optimizador de Next no puede procesar. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={original}
          alt="Radiografía original"
          onLoad={(e) =>
            setNatural({
              ancho: e.currentTarget.naturalWidth,
              alto: e.currentTarget.naturalHeight,
            })
          }
          className={claseImagen}
          style={transformacion}
        />

        {/* Capa recortada: el recorte ocurre en coordenadas de pantalla, de modo
            que el divisor no se mueve al desplazar o ampliar. */}
        <div
          className="absolute inset-0"
          style={{
            clipPath: `inset(0 ${explicacion ? 100 - division : 100}% 0 0)`,
            transition: "clip-path 0.3s ease",
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={capa === "superpuesta" ? superpuesta : mapa}
            alt="Mapa de atención Grad-CAM"
            className={claseImagen}
            style={transformacion}
          />
        </div>

        {/* Divisor: solo cuando la explicabilidad esta desplegada */}
        <div
          className={`sin-imprimir absolute inset-y-0 z-10 w-px bg-white/70 transition-opacity ${
            explicacion ? "opacity-100" : "pointer-events-none opacity-0"
          }`}
          style={{ left: `${division}%` }}
        >
          <button
            aria-label="Mover el divisor de comparación"
            onPointerDown={(e) => {
              e.stopPropagation();
              alPresionar(e, "divisor");
            }}
            onPointerMove={alMover}
            onPointerUp={alSoltar}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") setDivision((d) => Math.max(0, d - 2));
              if (e.key === "ArrowRight") setDivision((d) => Math.min(100, d + 2));
            }}
            className="absolute left-1/2 top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full border border-white/40 bg-black/70 text-white/90 backdrop-blur-sm transition hover:border-[var(--acento-vivo)]"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path
                d="M6 4L2.5 8 6 12M10 4l3.5 4-3.5 4"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>

        {/* Rotulos de cada lado */}
        {explicacion && (
          <>
            <span className="sin-imprimir pointer-events-none absolute bottom-3 left-3 rounded bg-black/60 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-white/80 backdrop-blur-sm">
              Original
            </span>
            <span className="sin-imprimir pointer-events-none absolute bottom-3 right-3 rounded bg-black/60 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-white/80 backdrop-blur-sm">
              Grad-CAM
            </span>
          </>
        )}
      </div>

      {/* Aclaracion imprescindible en los casos sin hallazgos: sin ella, el
          rojo se lee como patologia. */}
      {explicacion && !esAnomalia && (
        <p className="flex gap-2 border-t border-[var(--borde)] bg-[var(--estado-bien)]/8 px-4 py-3 text-xs leading-relaxed text-[var(--tinta-media)]">
          <span aria-hidden="true" className="text-[var(--estado-bien)]">
            ⓘ
          </span>
          <span>
            Este mapa <strong className="font-medium text-[var(--tinta)]">no señala
            hallazgos</strong>: marca las regiones en las que el modelo se apoyó para
            concluir que la radiografía es normal. El color intenso indica peso en la
            decisión, no anomalía.
          </span>
        </p>
      )}

      <figcaption className="border-t border-[var(--borde)] px-4 py-2.5 text-xs text-[var(--tinta-tenue)]">
        {explicacion ? "Arrastra el divisor para comparar. " : ""}Rueda del ratón para
        ampliar
        {natural.ancho > 0 && (
          <>
            {" "}
            · Original {natural.ancho} × {natural.alto} px
          </>
        )}
      </figcaption>
    </figure>
  );
}

function BotonHerramienta({
  children,
  onClick,
  desactivado,
  titulo,
}: {
  children: React.ReactNode;
  onClick: () => void;
  desactivado?: boolean;
  titulo: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={desactivado}
      title={titulo}
      aria-label={titulo}
      className="grid h-7 min-w-7 place-items-center rounded-md border border-[var(--borde)] px-1.5 text-sm text-[var(--tinta-media)] transition hover:border-[var(--borde-fuerte)] hover:text-[var(--tinta)] disabled:cursor-not-allowed disabled:opacity-35"
    >
      {children}
    </button>
  );
}
