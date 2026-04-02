# jocarsa | avatar

Visor 3D de avatar basado en **A-Frame** y **GLB**, preparado para cargar un personaje, iluminarlo en escena, orbitar la cámara alrededor del modelo y controlar en tiempo real la rotación del hueso **`cabeza`** mediante mensajes `postMessage()`.

Este proyecto está pensado como una base sencilla para demos, pruebas de rigging, control facial básico y futuras interfaces de animación para personajes 3D en web.

## Repositorio

`https://github.com/jocarsa/avatar`

## Demo en GitHub Pages

`https://jocarsa.github.io/avatar`

---

## Características

* Carga de modelos **GLB** mediante `gltf-model`
* Escena 3D con **A-Frame**
* Cámara con controles orbitales
* Iluminación ambiental y direccional
* Suelo receptor de sombras
* Detección automática del hueso llamado **`cabeza`**
* Control de rotación del hueso en los ejes **X**, **Y** y **Z**
* Comunicación externa mediante **`window.postMessage()`**
* Base simple y clara para extender con más huesos, morph targets o animaciones

---

## Estructura esperada

El proyecto puede funcionar con una estructura mínima como esta:

```text
avatar/
├── index.html
├── avatar.glb
└── README.md
```

---

## Requisitos

Solo necesitas:

* un navegador moderno
* un archivo `avatar.glb` en la misma carpeta que `index.html`

No hace falta compilación ni dependencias instaladas por npm para esta versión básica, ya que las librerías se cargan desde CDN.

---

## Uso local

Basta con servir la carpeta con un servidor web simple. Por ejemplo:

```bash
python3 -m http.server 8000
```

Y después abrir:

```text
http://localhost:8000
```

---

## Funcionamiento general

La escena:

* carga el modelo `avatar.glb`
* busca dentro del esqueleto un hueso con nombre exacto **`cabeza`**
* guarda la rotación base de ese hueso
* aplica sobre esa base una rotación adicional controlada externamente

De ese modo, el personaje conserva su pose original y recibe solo el desplazamiento adicional que se le envía.

---

## Control externo con `postMessage()`

La página escucha mensajes entrantes y admite dos tipos de órdenes.

### 1. Establecer rotación de la cabeza

```javascript
iframe.contentWindow.postMessage({
  type: 'setCabezaRotation',
  x: 10,
  y: -20,
  z: 5
}, '*');
```

### 2. Reiniciar rotación

```javascript
iframe.contentWindow.postMessage({
  type: 'resetCabezaRotation'
}, '*');
```

---

## Rangos aplicados

El código limita internamente los valores para evitar movimientos excesivos:

* **X:** de `-30` a `30`
* **Y:** de `-60` a `60`
* **Z:** de `-20` a `20`

---

## Requisito importante del rig

Para que el control funcione, el archivo GLB debe contener un hueso con este nombre exacto:

```text
cabeza
```

Si no existe, el visor no podrá mover la cabeza y mostrará en consola un aviso con la lista de huesos detectados.

---

## Tecnologías usadas

* **A-Frame 1.7.0**
* **aframe-orbit-controls-component**
* **Three.js** a través de A-Frame
* **GLTF / GLB**

---

## Ejemplo de integración

Este visor está preparado para ser embebido en un `iframe` desde otra página que actúe como panel de control.

Ejemplo:

```html
<iframe id="viewer" src="index.html"></iframe>

<script>
  const frame = document.getElementById('viewer');

  frame.contentWindow.postMessage({
    type: 'setCabezaRotation',
    x: 0,
    y: 25,
    z: 0
  }, '*');
</script>
```

Esto permite crear fácilmente:

* paneles de control de huesos
* interfaces de prueba de rig
* sistemas de animación manual
* demos educativas sobre esqueletos 3D
* prototipos de avatares interactivos

---

## Posibles ampliaciones

Este proyecto puede crecer fácilmente para incluir:

* control de más huesos
* morph targets faciales
* sincronización labial
* presets de poses
* animaciones automáticas
* captura de expresiones
* integración con chat o voz
* panel de administración estilo Jocarsa

---

## Publicación en GitHub Pages

Si el archivo `index.html` está en la raíz del repositorio y `avatar.glb` también, GitHub Pages puede publicar directamente el visor en:

```text
https://jocarsa.github.io/avatar
```

---

## Licencia

Puedes añadir aquí la licencia que prefieras para el proyecto, por ejemplo:

```text
MIT
```

---

## Autor

**Jose Vicente Carratala**
**Jocarsa**


