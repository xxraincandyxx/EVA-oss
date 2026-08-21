export const publicUrl = (path) =>
  `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`;
