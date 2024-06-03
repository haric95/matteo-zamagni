export const parseTagsString = (tags: string) => {
  return tags.split(",").map((tag) => tag.trim());
};
