/// Axis-aligned bounding box (AABB) representation.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct HalaBounds {
  pub center: [f32; 3],
  pub extents: [f32; 3],
}

/// Implementation of HalaBounds.
impl HalaBounds {

  /// Create a new HalaBounds instance.
  /// param center: The center of the AABB.
  /// param extents: The extents of the AABB.
  /// return: The new HalaBounds instance.
  pub fn new(center: [f32; 3], extents: [f32; 3]) -> Self {
    Self { center, extents }
  }

  /// Get the size of the AABB.
  /// return: The size of the AABB.
  pub fn get_size(&self) -> [f32; 3] {
    [self.extents[0] * 2.0, self.extents[1] * 2.0, self.extents[2] * 2.0]
  }

  /// Set the size of the AABB.
  /// param size: The new size of the AABB.
  pub fn set_size(&mut self, size: [f32; 3]) {
    self.extents = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
  }

  /// Get the minimum bounds of the AABB.
  /// return: The minimum bounds of the AABB.
  pub fn get_min(&self) -> [f32; 3] {
    [
      self.center[0] - self.extents[0],
      self.center[1] - self.extents[1],
      self.center[2] - self.extents[2],
    ]
  }

  /// Set the minimum bounds of the AABB.
  /// param min: The minimum bounds of the AABB.
  pub fn set_min(&mut self, min: [f32; 3]) {
    self.set_min_max(min, self.get_max());
  }

  /// Get the maximum bounds of the AABB.
  /// return: The maximum bounds of the AABB.
  pub fn get_max(&self) -> [f32; 3] {
    [
      self.center[0] + self.extents[0],
      self.center[1] + self.extents[1],
      self.center[2] + self.extents[2],
    ]
  }

  /// Set the maximum bounds of the AABB.
  /// param max: The maximum bounds of the AABB.
  pub fn set_max(&mut self, max: [f32; 3]) {
    self.set_min_max(self.get_min(), max);
  }

  /// Set the minimum and maximum bounds of the AABB.
  /// param min: The minimum bounds of the AABB.
  /// param max: The maximum bounds of the AABB.
  pub fn set_min_max(&mut self, min: [f32; 3], max: [f32; 3]) {
    self.extents = [
      (max[0] - min[0]) * 0.5,
      (max[1] - min[1]) * 0.5,
      (max[2] - min[2]) * 0.5,
    ];
    self.center = [
      min[0] + self.extents[0],
      min[1] + self.extents[1],
      min[2] + self.extents[2],
    ];
  }

  /// Grows the AABB to include the given point.
  /// param point: The point to include.
  pub fn encapsulate_point(&mut self, point: [f32; 3]) {
    let min = self.get_min();
    let max = self.get_max();
    self.set_min_max(
      [
        min[0].min(point[0]),
        min[1].min(point[1]),
        min[2].min(point[2]),
      ],
      [
        max[0].max(point[0]),
        max[1].max(point[1]),
        max[2].max(point[2]),
      ],
    );
  }

  /// Grows the AABB to include the given bounds.
  /// param bounds: The bounds to include.
  pub fn encapsulate_bounds(&mut self, bounds: &HalaBounds) {
    self.encapsulate_point([
      bounds.center[0] - bounds.extents[0],
      bounds.center[1] - bounds.extents[1],
      bounds.center[2] - bounds.extents[2],
    ]);
    self.encapsulate_point([
      bounds.center[0] + bounds.extents[0],
      bounds.center[1] + bounds.extents[1],
      bounds.center[2] + bounds.extents[2],
    ]);
  }

  /// Expands the AABB by the given amount.
  /// param amount: The amount to expand by.
  pub fn expand(&mut self, amount: f32) {
    let amount = amount * 0.5;
    self.extents[0] += amount;
    self.extents[1] += amount;
    self.extents[2] += amount;
  }

  /// Expands the AABB by the given amounts.
  /// param amounts: The amounts to expand by.
  pub fn expand_by(&mut self, amounts: [f32; 3]) {
    self.extents[0] += amounts[0] * 0.5;
    self.extents[1] += amounts[1] * 0.5;
    self.extents[2] += amounts[2] * 0.5;
  }

  /// Does another AABB intersect with this AABB?
  /// param other: The other AABB to check.
  /// return: True if the AABBs intersect, false otherwise.
  pub fn intersects(&self, other: &HalaBounds) -> bool {
    let min = self.get_min();
    let max = self.get_max();
    let other_min = other.get_min();
    let other_max = other.get_max();
    min[0] <= other_max[0] && max[0] >= other_min[0]
      && min[1] <= other_max[1] && max[1] >= other_min[1]
      && min[2] <= other_max[2] && max[2] >= other_min[2]
  }

}