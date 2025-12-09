export const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString();

export const langName = (code) => {
  const map = { en: "English", hi: "Hindi", es: "Spanish", fr: "French" };
  return map[code] || code || "unknown";
};
