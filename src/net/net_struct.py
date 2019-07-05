# For conv and fc layers, if specifying reg, reg=regularization_strength
# if using L1 regularization, reg_type='l1'

architectures = {
  '000': [
  ('conv', {'filter_size': 2,
            'num_filters': 64,
            'stride': 1}),
  ('conv', {'filter_size': 2,
            'num_filters': 96,
            'stride': 1,
            'padding': 'SAME'}),
  ('conv', {'filter_size': 2,
            'num_filters': 192,
            'stride': 1,
            'padding': 'SAME'}),
  ('pool', {'pooling_size': 2}),
  ('conv', {'filter_size': 2,
            'num_filters': 384,
            'stride': 2}),
  ('pool', {'pooling_size': 2}),
  ('conv', {'filter_size': 2,
            'num_filters': 768,
            'stride': 1}),
  ('conv', {'filter_size': 2,
            'num_filters': 2048,
            'stride': 1}),
  ('conv2fc', {}),
  ('fc', {'hidden_dim': 1024}),
  ('fc', {'hidden_dim': 512}),
  ('output', {'num_classes': 10})],

  '001': [
  ('conv', {'filter_size': 2,
            'num_filters': 64,
            'stride': 1}),
  ('conv', {'filter_size': 2,
            'num_filters': 96,
            'stride': 1,
            'padding': 'SAME'}),
  ('conv', {'filter_size': 2,
            'num_filters': 192,
            'stride': 1,
            'padding': 'SAME'}),
  ('pool', {'pooling_size': 2}),
  ('conv', {'filter_size': 2,
            'num_filters': 384,
            'stride': 2}),
  ('pool', {'pooling_size': 2}),
  ('conv', {'filter_size': 2,
            'num_filters': 768,
            'stride': 1}),
  ('conv', {'filter_size': 2,
            'num_filters': 2048,
            'stride': 1}),
  ('conv2fc', {}),
  ('fc', {'hidden_dim': 1024}),
  ('fc', {'hidden_dim': 512}),
  ('output', {'num_classes': 151})],
  
  }

def get_architecture(architecture_ID):
  return architectures[architecture_ID]
  