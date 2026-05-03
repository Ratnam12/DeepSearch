/**
 * Stubs the `server-only` package so library files that import it can
 * be loaded from a Node script (the package normally throws on
 * import, since it's a Next.js-time guard).
 *
 * Loaded via NODE_OPTIONS=--require=./scripts/server-only-shim.cjs by
 * the npm scripts that run our standalone smoke tests.
 */
// biome-ignore lint/suspicious/noShadowRestrictedNames: required for require.cache override
const Module = require("node:module");
const original = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === "server-only") {
    return require.resolve("./server-only-stub.cjs");
  }
  return original.call(this, request, ...rest);
};
